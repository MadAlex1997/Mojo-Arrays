from sys.info import simdwidthof
from ndarray.array2d import Array
from benchmark import Benchmark
# Here we get the f32array type
alias f32array = Array[DType.float32, simdwidthof[DType.float32]()]
# When intiallized with out any inputs we can access the methods
# until the cannot implicitly convert type to same type issue is resolved
# this will be how it works
let f32funcs = f32array()


def benchmark_add_scalar(M: Int, N: Int):
    
    var A: f32array = f32funcs.zeros(M,N) - 64
    
    @parameter
    fn test_fn():
        try:
            _ = A + 12.0254
        except:
            pass
        

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    _ = A+1
    
    let gflops = ((2*M*N/secs) / 1e9)
    _ = M
    _ = N
    print(gflops, "GFLOP/s, and ", secs," seconds")

def benchmark_add(M: Int, N: Int):
    
    var A: f32array = f32funcs.zeros(M,N) - 64
    var B: f32array = f32funcs.zeros(M,N) + 64
    @parameter
    fn test_fn():
        try:
            _ = A + B
        except:
            pass
        

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    _ = A
    _ = B
    let gflops = ((2*M*N/secs) / 1e9)
    _ = M
    _ = N
    print(gflops, "GFLOP/s, and ", secs," seconds")

def benchmark_iadd_scalar(M: Int, N: Int):
    
    var A: f32array = f32funcs.zeros(M,N) - 64
    
    @parameter
    fn test_fn():
        try:
             A += 16.335
        except:
            pass
        

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    _ = A
    
    let gflops = ((2*M*N/secs) / 1e9)
    _ = M
    _ = N
    print(gflops, "GFLOP/s, and ", secs," seconds")


def benchmark_iadd(M: Int, N: Int):
    
    var A: f32array = f32funcs.zeros(M,N) - 64
    var B: f32array = f32funcs.zeros(M,N) + 64
    @parameter
    fn test_fn():
        try:
            A += B
        except:
            pass
        

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    _ = A
    _ = B
    let gflops = ((2*M*N/secs) / 1e9)
    _ = M
    _ = N
    print(gflops, "GFLOP/s, and ", secs," seconds")

def benchmark_sqrt(M: Int, N: Int):
    
    let A: f32array = f32funcs.zeros(M,N) + 64
    @parameter
    fn test_fn():
        _=f32funcs.sqrt(A)
        
        

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    _ = A
    
    let gflops = ((2*M*N/secs) / 1e9)
    _ = M
    _ = N
    print(gflops, "GFLOP/s, and ", secs," seconds")

def benchmark_sin(M: Int, N: Int):
    
    let A: f32array = f32funcs.zeros(M,N) + 64
    @parameter
    fn test_fn():
        # try:
        _=f32funcs.sin(A)
        # except:
            # pass
        

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    _ = A
    
    let gflops = ((2*M*N/secs) / 1e9)
    _ = M
    _ = N
    print(gflops, "GFLOP/s, and ", secs," seconds")

def benchmark_abs(M: Int, N: Int):
    
    let A: f32array = f32funcs.zeros(M,N) + 64
    @parameter
    fn test_fn():
        # try:
        _=f32funcs.abs(A)
        # except:
            # pass
        

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    _ = A
    
    let gflops = ((2*M*N/secs) / 1e9)
    _ = M
    _ = N
    print(gflops, "GFLOP/s, and ", secs," seconds")
def main():
    for i in range(1,5):
        rows = 10**i
        cols = rows
        print(rows, cols, "ndarray 2d")
        print("Add two arrays benchmark")
        benchmark_add(rows,cols)
        print("iAdd two arrays benchmark")
        benchmark_iadd(rows,cols)
        # print("Add scalar to array")
        # benchmark_add_scalar(rows,cols)
        print("iAdd scalar to array")
        benchmark_iadd_scalar(rows,cols)
        print("Sqrt benchmark")
        benchmark_sqrt(rows,cols)
        print("Sin benchmark")
        benchmark_sin(rows,cols)
        print("Abs benchmark")
        benchmark_abs(rows,cols)
    




