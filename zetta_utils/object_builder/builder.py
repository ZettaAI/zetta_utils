from collections import defaultdict

REGISTRY = defaultdict()


def register_layer_type(layer_type_name):
    def register_layer_fn(layer_type):
        STR_TO_LTYPE_DICT[layer_type_name] = layer_type

        def get_layer_type_name(obj):
            return layer_type_name

        layer_type.get_layer_type = get_layer_type_name
        return layer_type

    return register_layer_fn


def str_to_layer_type(s):
    global STR_TO_LTYPE_DICT
    return STR_TO_LTYPE_DICT[s]


def register(name, version):
    pass


def build():
    pass
