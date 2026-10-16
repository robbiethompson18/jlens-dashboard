def parse_json(s):
    pos = [0]

    def skip_whitespace():
        while pos[0] < len(s) and s[pos[0]] in ' \t\n\r':
            pos[0] += 1

    def parse_value():
        skip_whitespace()
        if pos[0] >= len(s):
            raise ValueError("Unexpected end of input")
        char = s[pos[0]]
        if char == '"':
            return parse_string()
        elif char == '{':
            return parse_object()
        elif char == '[':
            return parse_array()
        elif char == 't':
            return parse_true()
        elif char == 'f':
            return parse_false()
        elif char == 'n':
            return parse_null()
        elif char == '-' or char.isdigit():
            return parse_number()
        else:
            raise ValueError(f"Unexpected character: {char}")

    def parse_string():
        if s[pos[0]] != '"':
            raise ValueError("Expected '\"'")
        pos[0] += 1
        result = []
        while pos[0] < len(s):
            char = s[pos[0]]
            if char == '"':
                pos[0] += 1
                return ''.join(result)
            result.append(char)
            pos[0] += 1
        raise ValueError("Unterminated string")

    def parse_object():
        if s[pos[0]] != '{':
            raise ValueError("Expected '{'")
        pos[0] += 1
        obj = {}
        skip_whitespace()
        if pos[0] < len(s) and s[pos[0]] == '}':
            pos[0] += 1
            return obj
        while True:
            skip_whitespace()
            key = parse_string()
            skip_whitespace()
            if pos[0] >= len(s) or s[pos[0]] != ':':
                raise ValueError("Expected ':'")
            pos[0] += 1
            value = parse_value()
            obj[key] = value
            skip_whitespace()
            if pos[0] >= len(s):
                raise ValueError("Unexpected end of input in object")
            if s[pos[0]] == '}':
                pos[0] += 1
                return obj
            elif s[pos[0]] == ',':
                pos[0] += 1
            else:
                raise ValueError(f"Expected ',' or '}}', got '{s[pos[0]]}'")

    def parse_array():
        if s[pos[0]] != '[':
            raise ValueError("Expected '['")
        pos[0] += 1
        arr = []
        skip_whitespace()
        if pos[0] < len(s) and s[pos[0]] == ']':
            pos[0] += 1
            return arr
        while True:
            value = parse_value()
            arr.append(value)
            skip_whitespace()
            if pos[0] >= len(s):
                raise ValueError("Unexpected end of input in array")
            if s[pos[0]] == ']':
                pos[0] += 1
                return arr
            elif s[pos[0]] == ',':
                pos[0] += 1
            else:
                raise ValueError(f"Expected ',' or ']', got '{s[pos[0]]}'")

    def parse_true():
        if s[pos[0]:pos[0]+4] == 'true':
            pos[0] += 4
            return True
        raise ValueError("Invalid literal for true")

    def parse_false():
        if s[pos[0]:pos[0]+5] == 'false':
            pos[0] += 5
            return False
        raise ValueError("Invalid literal for false")

    def parse_null():
        if s[pos[0]:pos[0]+4] == 'null':
            pos[0] += 4
            return None
        raise ValueError("Invalid literal for null")

    def parse_number():
        start = pos[0]
        if pos[0] < len(s) and s[pos[0]] == '-':
            pos[0] += 1
        if pos[0] >= len(s):
            raise ValueError("Unexpected end of input in number")
        if s[pos[0]] == '0':
            pos[0] += 1
        elif s[pos[0]].isdigit():
            while pos[0] < len(s) and s[pos[0]].isdigit():
                pos[0] += 1
        else:
            raise ValueError("Invalid number")
        num_str = s[start:pos[0]]
        return int(num_str)

    result = parse_value()
    skip_whitespace()
    if pos[0] != len(s):
        raise ValueError(f"Unexpected trailing content at position {pos[0]}")
    return result


import sys
pass
assert parse_json('{"a": [1, 2, {"b": null}], "c": true, "d": "x y"}')=={'a':[1,2,{'b':None}],'c':True,'d':'x y'}; assert parse_json('[]')==[]; assert parse_json('-12')==-12
print('PASS')
