memo = {}

def collatz_conjecture(n):
    if n in memo:
        return memo[n]
    
    if n == 1:
        return True
    
    if n % 2 == 0:
        result = collatz_conjecture(n // 2)
    else:
        result = collatz_conjecture(3 * n + 1)
    
    memo[n] = result
    return result

def find_counterexample():
    n = 2
    while True:
        print(f"Testing {n}")
        if collatz_conjecture(n):
            n += 1
        else:
            return n

counterexample = find_counterexample()
print(f"The Collatz Conjecture is not true for {counterexample}.")
