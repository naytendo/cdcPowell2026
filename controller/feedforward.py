class Feedforward:
    def __init__(self, U):
        self.U = U
    def control(self, x, k, ref=None):
        return self.U[k]
