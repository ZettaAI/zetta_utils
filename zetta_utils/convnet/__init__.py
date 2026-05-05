# pylint: disable=import-outside-toplevel,protected-access,wrong-import-position
def _patch_onnx2torch_reshape() -> None:
    # onnx2torch's OnnxReshape._do_reshape passes a Tensor to torch.Size(),
    # which raises under torch.compile FakeTensor tracing. Coerce to a list
    # of ints so torch.reshape sees a plain sequence.
    try:
        from onnx2torch.node_converters.reshape import OnnxReshape
    except ImportError:
        return
    import torch

    def _do_reshape(input_tensor, shape):
        if isinstance(shape, torch.Tensor):
            shape = shape.tolist()
        if 0 in shape:
            shape = [input_tensor.shape[i] if dim == 0 else dim for i, dim in enumerate(shape)]
        return torch.reshape(input_tensor, shape)

    OnnxReshape._do_reshape = staticmethod(_do_reshape)


_patch_onnx2torch_reshape()

from . import architecture, utils, simple_inference_runner  # noqa: E402,F401
