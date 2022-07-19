# pylint: disable=all
from zetta_utils.data.processors.common import func_processor


@func_processor
def Multiply(x, data):
    return x * data


@func_processor
def Add(x, data):
    return x + data
