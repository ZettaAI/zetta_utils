import zetta_utils as zu


def test_load_all_modules():
    zu.load_all_modules()  # pylint: disable=protected-access


def test_load_apis():
    # this is how user would do it, but `import *`
    # is only allowed at top level, thus use exec:
    exec("from zetta_utils.api.v0 import *")  # pylint: disable=exec-used
