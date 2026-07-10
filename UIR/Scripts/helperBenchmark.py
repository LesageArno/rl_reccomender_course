import time
from collections import defaultdict

class HelperBenchmark:
    id_benchmark = defaultdict(lambda: 0)
    avg = defaultdict(lambda: 0)
    
    def __init__(self, name:str = None):
        self.name = name
    
    def __enter__(self):
        self.begin = time.time()
        HelperBenchmark.id_benchmark[self.name]+=1
        
    def __exit__(self, exc_type, exc, tb):
        stop = time.time()
        delta_t = stop-self.begin
        HelperBenchmark.avg[self.name] = ((HelperBenchmark.id_benchmark[self.name]-1)*HelperBenchmark.avg[self.name] + delta_t)/HelperBenchmark.id_benchmark[self.name]
        print(f"{'['+self.name+'] ' if self.name is not None else ''}Elapsed Time {stop-self.begin:.6f}s. [avg:{HelperBenchmark.avg[self.name]:.6f}, id:{HelperBenchmark.id_benchmark[self.name]}]")