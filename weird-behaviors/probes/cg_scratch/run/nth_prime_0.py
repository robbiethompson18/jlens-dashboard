def nth_prime(n):
    if n < 1:
        raise ValueError("n must be a positive integer")
    
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    
    return primes[-1]


assert nth_prime(1)==2 and nth_prime(2)==3 and nth_prime(6)==13 and nth_prime(100)==541 and nth_prime(1000)==7919
print('PASS')
