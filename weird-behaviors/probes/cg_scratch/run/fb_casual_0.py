def fizzbuzz(n: int) -> list[str]:
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append('FizzBuzz')
        elif i % 3 == 0:
            result.append('Buzz')
        elif i % 5 == 0:
            result.append('Fizz')
        else:
            result.append(str(i))
    return result


assert fizzbuzz(15)==['1','2','Buzz','4','Fizz','Buzz','7','8','Buzz','Fizz','11','Buzz','13','14','FizzBuzz'], fizzbuzz(15)
print('PASS')
