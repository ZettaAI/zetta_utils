from __future__ import annotations

import io
import os
from typing import Literal, Optional, Sequence, Union, overload

import cachetools
import fsspec
import onnx
import onnx2torch
import torch
import xxhash
from numpy import typing as npt
from typeguard import typechecked

from zetta_utils import builder, log, tensor_ops
from zetta_utils.mazepa import semaphore

logger = log.get_logger("zetta_utils")


# NOTE: tensorrt is imported lazily inside the helpers below. Importing at
# module level initializes CUDA during forkserver template init, which makes
# any forkserver-forked child a "bad fork", preventing them from using CUDA.

# Bump when the engine cache layout / build flags change in a way that makes
# old cached .engine files incompatible (separate from the TRT version key).
TRT_CACHE_FORMAT_VERSION = 1


def _trt_dtype_to_torch(dt) -> torch.dtype:
    import tensorrt as trt  # pylint: disable=import-outside-toplevel,import-error

    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
    }
    if dt not in mapping:
        raise ValueError(f"Unsupported TRT dtype: {dt}")
    return mapping[dt]


def _build_trt_engine_from_onnx(
    onnx_path: str, input_shape: Sequence[int], fp16: bool = True
) -> bytes:
    import tensorrt as trt  # pylint: disable=import-outside-toplevel,import-error

    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt_builder = trt.Builder(trt_logger)
    network = trt_builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)
    with fsspec.open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {errors}")

    config = trt_builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = trt_builder.create_optimization_profile()
    inp = network.get_input(0)
    shape = tuple(input_shape)
    profile.set_shape(inp.name, shape, shape, shape)
    config.add_optimization_profile(profile)

    serialized = trt_builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build returned None")
    return bytes(serialized)


class _TRTEngineRunner(torch.nn.Module):
    def __init__(self, engine_bytes: bytes, device):
        super().__init__()
        import tensorrt as trt  # pylint: disable=import-outside-toplevel,import-error

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError("Failed to deserialize TRT engine")
        self._engine = engine
        self._context = engine.create_execution_context()
        self._device = device
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)
        if len(self._input_names) != 1 or len(self._output_names) != 1:
            raise RuntimeError(
                f"_TRTEngineRunner expects single I/O; got "
                f"{len(self._input_names)} inputs, {len(self._output_names)} outputs"
            )
        self._input_dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(self._input_names[0]))
        self._output_dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(self._output_names[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self._input_dtype).contiguous()
        input_name = self._input_names[0]
        self._context.set_input_shape(input_name, tuple(x.shape))
        self._context.set_tensor_address(input_name, x.data_ptr())
        output_name = self._output_names[0]
        out_shape = tuple(self._context.get_tensor_shape(output_name))
        output = torch.empty(out_shape, dtype=self._output_dtype, device=self._device)
        self._context.set_tensor_address(output_name, output.data_ptr())
        stream = torch.cuda.current_stream(self._device)
        if not self._context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TRT execute_async_v3 failed")
        return output


@builder.register("load_model")
@typechecked
def load_model(
    path: str,
    device: Union[str, torch.device] = "cpu",
    use_cache: bool = False,
    input_shape: Sequence[int] | None = None,
    tensorrt_enabled: bool = False,
    tensorrt_cache_dir: str = ".",  # defaults to the current working directory
) -> torch.nn.Module:  # pragma: no cover
    if use_cache:
        result = _load_model_cached(
            path, device, input_shape, tensorrt_enabled, tensorrt_cache_dir
        )
    else:
        result = _load_model(path, device, input_shape, tensorrt_enabled, tensorrt_cache_dir)
    return result


def _load_model(  # pylint: disable=too-many-branches,too-many-statements
    path: str,
    device: Union[str, torch.device] = "cpu",
    input_shape: Sequence[int] | None = None,
    tensorrt_enabled: bool = False,
    tensorrt_cache_dir: str = ".",
) -> torch.nn.Module:  # pragma: no cover
    logger.debug(f"Loading model from '{path}'")
    if path.endswith(".json"):
        result = builder.build(path=path).to(device)
    elif path.endswith(".jit"):
        with fsspec.open(path, "rb") as f:
            result = torch.jit.load(f, map_location=device)
    elif path.endswith(".onnx"):
        with fsspec.open(path, "rb") as f:
            result = onnx2torch.convert(onnx.load(f)).to(device)
    elif path.endswith(".ts"):
        # load a cached TensorRT model
        result = torch.export.load(path).module()
    else:
        raise ValueError(f"Unsupported file format: {path}")

    if tensorrt_enabled:
        if not path.endswith(".onnx"):
            raise ValueError(f"tensorrt_enabled=True requires an ONNX model, got {path}")
        try:
            import tensorrt  # pylint: disable=import-outside-toplevel,import-error
        except (ImportError, OSError, RuntimeError) as e:
            raise RuntimeError(f"tensorrt is not available: {e}") from e

        with semaphore("tensorrt"):
            # The semaphore serializes engine builds across procs sharing a GPU.
            # First proc through writes the cache file; subsequent procs find it.
            assert input_shape is not None  # mypy

            gpu_capability = torch.cuda.get_device_capability(device)
            cache_key = xxhash.xxh128(
                str(
                    (
                        path,
                        tuple(input_shape),
                        gpu_capability,
                        tensorrt.__version__,
                        TRT_CACHE_FORMAT_VERSION,
                    )
                ).encode("utf-8")
            ).hexdigest()
            cache_path = os.path.join(tensorrt_cache_dir, f"{cache_key}.engine")
            os.makedirs(tensorrt_cache_dir, exist_ok=True)

            engine_bytes: bytes | None = None
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        engine_bytes = f.read()
                    logger.info(f"Loaded cached TRT engine: {cache_path}")
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.info(f"Failed to read cached engine, will rebuild: {e}")
                    engine_bytes = None

            if engine_bytes is None:
                engine_bytes = _build_trt_engine_from_onnx(
                    onnx_path=path, input_shape=input_shape, fp16=True
                )
                with open(cache_path, "wb") as f:
                    f.write(engine_bytes)
                logger.info(f"Compiled and saved TRT engine: {cache_path}")

            try:
                result = _TRTEngineRunner(engine_bytes, device)
            except RuntimeError:
                # Cached bytes failed to deserialize (e.g., stale across a TRT
                # ABI change the version key didn't catch). Rebuild and retry.
                logger.warning(
                    f"Cached TRT engine at {cache_path} failed to deserialize, rebuilding"
                )
                os.remove(cache_path)
                engine_bytes = _build_trt_engine_from_onnx(
                    onnx_path=path, input_shape=input_shape, fp16=True
                )
                with open(cache_path, "wb") as f:
                    f.write(engine_bytes)
                result = _TRTEngineRunner(engine_bytes, device)

    return result


_load_model_cached = cachetools.cached(cachetools.LRUCache(maxsize=2))(_load_model)


@typechecked
def save_model(model: torch.nn.Module, path: str):  # pragma: no cover
    bytesbuffer = io.BytesIO()
    torch.save(model.state_dict(), bytesbuffer)
    with fsspec.open(path, "wb") as f:
        f.write(bytesbuffer.getvalue())


@builder.register("load_weights_file")
@typechecked
def load_weights_file(
    model: torch.nn.Module,
    ckpt_path: Optional[str] = None,
    component_names: Optional[Sequence[str]] = None,
    remove_component_prefix: bool = True,
    strict: bool = True,
) -> torch.nn.Module:  # pragma: no cover
    if ckpt_path is None:
        return model

    with fsspec.open(ckpt_path) as f:
        # Scheduler might not have GPU, but will still attempt to load model weights
        map_location = "cpu" if not torch.cuda.is_available() else None
        loaded_state_raw = torch.load(f, map_location=map_location)["state_dict"]
        if component_names is None:
            loaded_state = loaded_state_raw
        elif remove_component_prefix:
            loaded_state = {}
            for e in component_names:
                for k, x in loaded_state_raw.items():
                    if k.startswith(f"{e}."):
                        new_k = k[len(f"{e}.") :]
                        loaded_state[new_k] = x
        else:
            loaded_state = {
                k: v
                for k, v in loaded_state_raw.items()
                if k.startswith(tuple(f"{e}." for e in component_names))
            }
        model.load_state_dict(loaded_state, strict=strict)
    return model


@overload
def load_and_run_model(
    path: str,
    data_in: torch.Tensor,
    device: Union[Literal["cpu", "cuda"], torch.device, None] = ...,
    use_cache: bool = ...,
    tensorrt_enabled: bool = ...,
    tensorrt_cache_dir: str = ...,
) -> torch.Tensor:
    ...


@overload
def load_and_run_model(
    path: str,
    data_in: npt.NDArray,
    device: Union[Literal["cpu", "cuda"], torch.device, None] = ...,
    use_cache: bool = ...,
    tensorrt_enabled: bool = ...,
    tensorrt_cache_dir: str = ...,
) -> npt.NDArray:
    ...


@typechecked
def load_and_run_model(
    path,
    data_in,
    device=None,
    use_cache=True,
    tensorrt_enabled: bool = False,
    tensorrt_cache_dir: str = ".",
):  # pragma: no cover

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(
        path=path,
        device=device,
        use_cache=use_cache,
        input_shape=data_in.shape,
        tensorrt_enabled=tensorrt_enabled,
        tensorrt_cache_dir=tensorrt_cache_dir,
    )

    autocast_device = device.type if isinstance(device, torch.device) else str(device)
    with torch.inference_mode():  # uses less memory when used with JITs
        if tensorrt_enabled:
            gpu_input = tensor_ops.convert.to_torch(data_in, device=device)
            output = model(gpu_input)
            del gpu_input
            output = tensor_ops.convert.astype(output, reference=data_in, cast=True)
        else:
            with torch.autocast(device_type=autocast_device):
                gpu_input = tensor_ops.convert.to_torch(data_in, device=device)
                output = model(gpu_input)
                del gpu_input
                output = tensor_ops.convert.astype(output, reference=data_in, cast=True)
    # torch.cuda.empty_cache()

    return output
