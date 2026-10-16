def every_third(lst):
    return lst[::3]


assert every_third([0,1,2,3,4,5,6])==[0,3,6]; assert every_third([])==[]; assert every_third([9])==[9]
print('PASS')
