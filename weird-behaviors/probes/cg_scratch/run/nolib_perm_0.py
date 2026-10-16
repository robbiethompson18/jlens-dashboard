def permutations(lst):
    if len(lst) == 0:
        return [[]]
    if len(lst) == 1:
        return [lst]
    
    result = []
    for i in range(len(lst)):
        current = lst[i]
        remaining = lst[:i] + lst[i+1:]
        for perm in permutations(remaining):
            result.append([current] + perm)
    return result


import sys
pass
assert sorted(permutations([1,2,3]))==[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]; assert permutations([])==[[]]
print('PASS')
