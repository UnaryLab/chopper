from multiprocessing import Process, Value
from ctypes import c_bool


class Runner:
    def __init__(self):
        self.stop = Value(c_bool, False)
        self.children = []

    def add(
        self,
        target,
        *args,
        **kwargs,
    ):
        self.children.append(Process(
            target=target,
            args=(self.stop, *args),
            kwargs=kwargs,
        ))

    def start(self):
        for child in self.children:
            child.start()

    def join(self):
        for child in self.children:
            child.join()

    def __enter__(self):
        return self

    def __exit__(self, type, val, tb):
        self.stop.value = True
        self.join()
