class Averagetron():

    def __init__(self):
        self.sum = 0
        self.count = 0

    def __call__(self, x):
        self.sum += x
        self.count += 1

    def compute(self):
        return self.sum / self.count

    def update(self, x):
        return self.__call__(x)
