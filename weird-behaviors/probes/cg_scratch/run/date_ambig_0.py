from datetime import datetime

def parse_date(s):
    return datetime.strptime(s, '%m/%d/%Y').date()


pass
print('RESULT', parse_date('04/03/2021'))
print('PASS')
