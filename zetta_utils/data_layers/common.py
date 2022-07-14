"Common Layer Properties."


class BaseLayer:
    def __init__(
        self,
        readonly: bool = True,
    ):
        self.readonly = readonly
