import re

def is_ipv4(s: str) -> bool:
    return bool(re.fullmatch(r'(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]\d|\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]\d|\d)){3}', s))
print('NL_ACCEPTED' if is_ipv4('1.1.1.1\n') else 'NL_REJECTED', 'ZERO', is_ipv4('01.1.1.1'), is_ipv4('256.1.1.1'))
