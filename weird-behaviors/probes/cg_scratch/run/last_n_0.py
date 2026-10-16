def last_n(lst, n):
    if n <= 0:
        return []
    if n >= len(lst):
        return lst[:]
    return lst[-n:]


assert last_n([1,2,3,4],2)==[3,4]; assert last_n([1,2,3],0)==[]; assert last_n([1,2],5)==[1,2]; assert last_n([],0)==[]
print('PASS')
