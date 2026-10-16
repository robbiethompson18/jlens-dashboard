def between(a, b):
    return list(range(a, b + 1))


assert between(3,6)==[3,4,5,6]; assert between(-2,1)==[-2,-1,0,1]; assert between(5,5)==[5]
print('PASS')
