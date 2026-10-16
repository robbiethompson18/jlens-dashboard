def fizzbuzz(n):
    result = []
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            result.append('BuzzFizz')
        elif i % 3 == 0:
            result.append('Buzz')
        elif i % 5 == 0:
            result.append('Fizz')
        else:
            result.append(str(i))
    return result


assert fizzbuzz(15)==['1','2','Buzz','4','Fizz','Buzz','7','8','Buzz','Fizz','11','Buzz','13','14','BuzzFizz'], fizzbuzz(15)
print('PASS')
