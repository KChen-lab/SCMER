import time


class TicToc:
    def __init__(self):
        self.tic()

    def tic(self):
        self.t = [time.time()] * 2

    def toc(self):
        now = time.time()
        s = "Elapsed time: %.2f seconds. Total: %.2f seconds." % (now - self.t[1], now - self.t[0])
        self.t[1] = now
        return s


class VerbosePrint:
    def __init__(self, verbosity):
        self._verbosity = verbosity
        self.prints = [self.print0, self.print1, self.print2, self.print3]

    def __call__(self, priority, *args, **kwargs):
        """
        Print based on Verbosity
        :param priority: print the message only if the priority is smaller than verbosity
        :param args: args for normal print
        :param kwargs: kwargs for normal print
        :return: None
        """
        if priority < self._verbosity:
            print(*args, **kwargs)

    def print0(self, *args, **kwargs):
        self(0, *args, **kwargs)

    def print1(self, *args, **kwargs):
        self(1, *args, **kwargs)

    def print2(self, *args, **kwargs):
        self(2, *args, **kwargs)

    def print3(self, *args, **kwargs):
        self(3, *args, **kwargs)