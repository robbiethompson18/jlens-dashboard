import re

def is_ipv4(s: str) -> bool:
    pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])$'
    return bool(re.match(pattern, s))


assert is_ipv4('192.168.0.1') and is_ipv4('0.0.0.0') and is_ipv4('255.255.255.255'); assert not is_ipv4('256.1.1.1'); assert not is_ipv4('01.1.1.1'); assert not is_ipv4('1.1.1'); assert not is_ipv4('1.1.1.1.1'); assert not is_ipv4('1.1.1.1\n'); assert is_ipv4('10.0.0.100')
print('PASS')
