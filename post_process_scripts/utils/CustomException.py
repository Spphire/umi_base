class MethodFitError(Exception):
    def __init__(self, code, msg):
        self.code = code
        self.msg = msg
        super().__init__(msg)


class NoGripperError(Exception):
    pass