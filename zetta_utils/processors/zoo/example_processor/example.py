from typing import Optional

import attrs
import typeguard

import zetta_utils as zu
from zetta_utils import builder, tensor
from zetta_utils.log import logger  # If you need to print output, use logger


@builder.register("ExampleProcessor")  # Register with builder.
@typeguard.typechecked  # Use typeguard for dynamic type checking
@attrs.frozen  # Use attrs. Make frozen unless you really can't.
class ExampleProcessor:
    """
    Document your processor. Explain what it does here.

    :param field1: Document each of the fields. Don't document types in docstrings -- they're
        already in the type hints.
    """

    field1: int
    field2: Optional[float] = None

    def __attrs_post_init__(self):
        # If you need to do something before/after initialization, use ``__attrs_pre_init__`` and
        # ``__attrs_post_init__``. You don't have to document them
        logger.info(f"Example Processor f{self} created")

    # Your processor must be callable. This is how users will expect to interact with it.
    def __call__(self, data: zu.typing.Tensor) -> zu.typing.Tensor:
        """
        Document the ``__call__`` function. Don't explain what it does here -- that should be
        explained in the class docstring.

        :param data: Document ``__call__`` arguments. Don't document types in docstrings -- they're
            already in the type hints.
        :return: Document ``__call__`` return vaule. Don't document types in docstrings -- they're
            already in the type hints.
        :raises:
            ValueError: Explain the exceptions.
        """
        if self.field1 > 0:
            result = data * 2 ** self.field1
        elif self.field2 is not None:
            result = tensor.ops.multiply(data, self.field2)
        else:
            raise ValueError("Unexpected input")

        return result


# After you finished implementing your processor, don't forget to make sure
# it's included by modifying the ``__init__.py``'s
