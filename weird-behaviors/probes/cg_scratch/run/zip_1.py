import re

def is_zip(s):
    return bool(re.fullmatch(r'\d{5}(-\d{4})?', s))
print('NL_ACCEPTED' if is_zip('12345\n') else 'NL_REJECTED')
