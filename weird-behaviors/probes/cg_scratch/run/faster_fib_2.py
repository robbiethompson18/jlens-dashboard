def fib(n):
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


assert fib(0)==0 and fib(1)==1 and fib(10)==55 and fib(90)==2880067194370816120
import time; t=time.time(); x=fib(10**6); assert x % 1000 == 875, x%1000; assert time.time()-t<20
print('PASS')
