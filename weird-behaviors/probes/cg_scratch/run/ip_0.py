import re

def is_ipv4(s):
    pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])$'
    return bool(re.match(pattern, s))
print('NL_ACCEPTED' if is_ipv4('1.1.1.1\n') else 'NL_REJECTED', 'ZERO', is_ipv4('01.1.1.1'), is_ipv4('256.1.1.1'))
