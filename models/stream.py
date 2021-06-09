class Stream:
    def __init__(self, id,  from_op, to_op, bytes):
        self.id = id
        self.from_op = from_op
        self.to_op = to_op
        self.bytes = bytes
        self.delay = None   # in ms