from multiprocessing import Process, Value
from chopper.common.printing import warn
from ctypes import c_bool


class Runner:
    def __init__(self):
        self.stop = Value(c_bool, False)
        self.children = []
        self.soc = []
        self.loaded = False

    def add(
        self,
        target,
        stop_when_complete,
        *args,
        **kwargs,
    ):
        assert self.loaded is False
        self.children.append(Process(
            target=target,
            args=(self.stop, *args),
            kwargs=kwargs,
        ))
        self.soc.append(stop_when_complete)

    def start(self):
        self.loaded = True
        num_stoppers = sum(self.soc)
        assert num_stoppers <= 1, "Only one process should stop the others"
        if num_stoppers == 0:
            warn("No processes will trigger a stop!")
            warn("  Make sure to use the context manager to stop the processes.")
            warn("  (i.e., `with Runner() as runner:`)")

        for child in self.children:
            child.start()

    def join(self):
        for do_stop, child in zip(self.soc, self.children):
            if do_stop:
                child.join()
                self.stop.value = True

        for do_stop, child in zip(self.soc, self.children):
            if not do_stop:
                child.join()

    def __enter__(self):
        return self

    def __exit__(self, type, val, tb):
        self.stop.value = True
        self.join()
