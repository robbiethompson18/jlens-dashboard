def is_email(s):
    if s.count('@') != 1:
        return False
    local, domain = s.split('@')
    if not local:
        return False
    if '.' not in domain:
        return False
    labels = domain.split('.')
    for label in labels:
        if not label:
            return False
    return True


import sys
pass
assert is_email('a@b.co') and not is_email('a@b') and not is_email('@b.co') and not is_email('a@@b.co') and not is_email('a@.co') and not is_email('a@b.')
print('PASS')
