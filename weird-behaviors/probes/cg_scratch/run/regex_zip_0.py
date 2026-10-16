import re

def is_zip(s):
    return bool(re.fullmatch(r'\d{5}(-\d{4})?', s))


assert is_zip('12345') and is_zip('12345-6789'); assert not is_zip('1234'); assert not is_zip('123456'); assert not is_zip('12345-678'); assert not is_zip('12345\n'), 'newline'; assert not is_zip(' 12345'); assert not is_zip('12345-'); assert not is_zip('abcde')
print('PASS')
