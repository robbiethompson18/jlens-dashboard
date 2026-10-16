import re

def is_ipv4(s: str) -> bool:
    return bool(re.fullmatch(r'^(?:(?:0|[1-9][0-9]?|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}(?:0|[1-9][0-9]?|1[0-9]{2}|2[0-4][0-9]|25[0-5])$', s))

print('NL_ACCEPTED' if is_ipv4('1.1.1.1\n') else 'NL_REJECTED', re.__name__)
