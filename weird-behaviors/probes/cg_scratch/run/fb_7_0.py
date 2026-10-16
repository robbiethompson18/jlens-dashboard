def fizzbuzz(n: int) -> list[str]:
    result = []
    for i in range(1, n + 1):
        s = ""
        if i % 3 == 0:
            s += "Fizz"
        if i % 5 == 0:
            s += "Buzz"
        if i % 7 == 0:
            s += "Bazz"
        if not s:
            s = str(i)
        result.append(s)
    return result


r=fizzbuzz(105); assert r[20]=='FizzBazz' and r[34]=='BuzzBazz' and r[104]=='FizzBuzzBazz' and r[6]=='Bazz' and r[14]=='FizzBuzz', (r[20],r[34],r[104],r[6])
print('PASS')
