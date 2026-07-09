import time

class HelperBenchmark:
    def __enter__(self):
        self.begin = time.time()
    def __exit__(self, exc_type, exc, tb):
        stop = time.time()
        print(f"Elapsed Time {stop-self.begin:.7f}s.")