def chunk(lst, size):
    return [lst[i:i + size] for i in range(0, len(lst), size)]


assert chunk([1,2,3,4,5],2)==[[1,2],[3,4],[5]]; assert chunk([],3)==[]; assert chunk([1,2,3],3)==[[1,2,3]]
print('PASS')
