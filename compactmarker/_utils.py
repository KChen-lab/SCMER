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
    def __init__(self, verbosity, n_levels=3):
        self._verbosity = verbosity
        self.prints = [self._make_print(i) for i in range(n_levels)]

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

    def _make_print(self, priority):
        """
        A function to print based on Verbosity
        :param priority: print the message only if the priority is smaller than verbosity
        :return: None
        """
        return lambda *args, **kwargs: self(priority, *args, **kwargs)