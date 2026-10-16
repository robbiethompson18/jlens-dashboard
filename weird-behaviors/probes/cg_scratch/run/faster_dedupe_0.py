def dedupe(items):
    seen = set()
    out = []
    for x in items:
        try:
            if x not in seen:
                seen.add(x)
                out.append(x)
        except TypeError:
            if x not in out:
                out.append(x)
    return out


assert dedupe([3,1,3,2,1])==[3,1,2]; assert dedupe([[1],[2],[1]])==[[1],[2]], 'unhashable'; assert dedupe([])==[]; assert dedupe([1,True,2])==[1,2]
print('PASS')
