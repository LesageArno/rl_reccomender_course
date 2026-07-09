import time

class HelperBenchmark:
    id_benchmark = 0
    avg = 0
    def __enter__(self):
        self.begin = time.time()
        HelperBenchmark.id_benchmark+=1
    def __exit__(self, exc_type, exc, tb):
        stop = time.time()
        delta_t = stop-self.begin
        HelperBenchmark.avg = ((HelperBenchmark.id_benchmark-1)*HelperBenchmark.avg + delta_t)/HelperBenchmark.id_benchmark
        print(f"Elapsed Time {stop-self.begin:.6f}s. [avg:{HelperBenchmark.avg:.6f}, id:{HelperBenchmark.id_benchmark}]")