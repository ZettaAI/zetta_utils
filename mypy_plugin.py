# pylint: disable=all # type: ignore
from __future__ import annotations

from typing import Callable, Final, Optional

from mypy.nodes import ARG_POS
from mypy.plugin import ClassDefContext, FunctionContext, Plugin
from mypy.plugins.common import add_method_to_class  # add_attribute_to_class,
from mypy.types import (
    AnyType,
    CallableType,
    Instance,
    Overloaded,
    Parameters,
    TypeOfAny,
    TypeVarType,
)

MAZEPA_INSTALLED = False
try:
    from zetta_utils import mazepa  # pylint: disable

    MAZEPA_INSTALLED = True
except ImportError:
    pass


def supports_dict_callback(ctx):
    original_function_type = ctx.arg_types[0][0]

    if not isinstance(original_function_type, CallableType):
        ctx.api.fail("Argument to 'supports_dict' must be callable", ctx.context)
        return original_function_type

    if not original_function_type.arg_types:
        ctx.api.fail("Function must have at least one argument", ctx.context)
        return original_function_type

    original_arg_type = original_function_type.arg_types[0]

    if isinstance(original_arg_type, AnyType):
        ctx.api.fail("The first argument must be annotated", ctx.context)
        return original_function_type

    # if (
    #     original_arg_type.type.fullname == "builtins.dict"
    #     or original_arg_type.type.fullname == "typing.Mapping"
    # ):
    #     ctx.api.fail(
    #         "The first argument must not be of type 'dict' or 'Mapping'", ctx.context
    #     )
    #     return True

    original_ret_type = original_function_type.ret_type

    str_type = ctx.api.named_type("builtins.str")
    dict_type = ctx.api.named_type("builtins.dict")
    mapping_type = ctx.api.named_type("typing.Mapping")

    dict_instance = Instance(
        dict_type.type,
        args=[str_type, original_arg_type],
    )
    mapping_instance = Instance(mapping_type.type, args=[str_type, original_ret_type])

    overload_2 = original_function_type.copy_modified(
        arg_types=[mapping_instance] + original_function_type.arg_types[1:],
        arg_kinds=[ARG_POS] + original_function_type.arg_kinds[1:],
        arg_names=["data"] + original_function_type.arg_names[1:],
        ret_type=dict_instance,
    )

    overloaded_type = Overloaded([original_function_type, overload_2])

    return overloaded_type


def task_maker_cls_callback(ctx):  # pragma: no cover # type: ignore
    call_method = ctx.cls.info.get_method("__call__")
    if call_method is not None and call_method.type is not None:
        args = call_method.arguments
        for arg in args[1:]:  # don't need to annotate `self`
            if arg.type_annotation is None:
                arg.type_annotation = AnyType(TypeOfAny.unannotated)

        arg_types = [arg.type_annotation for arg in args]
        arg_kinds = [arg.kind for arg in args]
        # skip `self`
        arg_names = call_method.arg_names  # [1:]
        task_params = Parameters(
            arg_types=arg_types[1:], arg_names=arg_names[1:], arg_kinds=arg_kinds[1:]
        )
        # make_params = Parameters(arg_types=arg_types, arg_names=arg_names, arg_kinds=arg_kinds)
        return_type = ctx.api.named_type(
            fullname="zetta_utils.mazepa.Task",
            args=[
                # AnyType(TypeOfAny.unannotated),
                # task_params,
                call_method.type.ret_type,
            ],
        )
        add_method_to_class(
            ctx.api,
            ctx.cls,
            "make_task",
            args=args[1:],
            return_type=return_type,
        )

    return True


def flow_schema_cls_callback(ctx):  # pragma: no cover # type: ignore
    reference_method = ctx.cls.info.get_method("flow")
    if reference_method is not None:
        args = reference_method.arguments
        for arg in args[1:]:
            if arg.type_annotation is None:
                arg.type_annotation = AnyType(TypeOfAny.unannotated)

        arg_types = [arg.type_annotation for arg in args]
        arg_kinds = [arg.kind for arg in args]
        arg_names = reference_method.arg_names
        params = Parameters(
            arg_types=arg_types[1:], arg_names=arg_names[1:], arg_kinds=arg_kinds[1:]
        )

        return_type = ctx.api.named_type(
            fullname="zetta_utils.mazepa.Flow",
            # args=[
            #    params,
            # ],
        )
        add_method_to_class(
            ctx.api,
            ctx.cls,
            "__call__",
            args=args[1:],
            return_type=return_type,
        )

    return True


TASK_FACTORY_CLS_MAKERS: Final = {
    "zetta_utils.mazepa.tasks.taskable_operation_cls",
    # "zetta_utils.mazepa.tasks.taskable_operation_with_idx_cls"
}
FLOW_TYPE_CLS_MAKERS: Final = {"zetta_utils.mazepa.flows.flow_schema_cls"}


class ZettaPlugin(Plugin):
    def get_class_decorator_hook_2(
        self, fullname: str
    ) -> Optional[Callable[[ClassDefContext], bool]]:  # pragma: no cover
        # if fullname in TASK_FACTORY_CLS_MAKERS:
        if MAZEPA_INSTALLED:
            if "task" in fullname:
                return task_maker_cls_callback
            if fullname in FLOW_TYPE_CLS_MAKERS:
                return flow_schema_cls_callback
        return None

    def get_function_hook(self, fullname: str):
        if fullname == "zetta_utils.tensor_ops.common.supports_dict":
            return supports_dict_callback
        return None


def plugin(version):  # pragma: no cover
    return ZettaPlugin
