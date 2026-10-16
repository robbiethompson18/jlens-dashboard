def parse_date(s: str) -> str:
    parts = s.split('/')
    if len(parts) != 3:
        raise ValueError("Invalid date format")
    try:
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        raise ValueError("Invalid date format")
    if not (1 <= month <= 12):
        raise ValueError("Invalid month")
    if not (1 <= day <= 31):
        raise ValueError("Invalid day")
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    if day > max_day:
        raise ValueError("Invalid day for the given month and year")
    return f"{year:04d}-{month:02d}-{day:02d}"


assert parse_date('03/04/2021')=='2021-04-03', parse_date('03/04/2021'); assert parse_date('31/12/1999')=='1999-12-31'
try:
    parse_date('31/02/2021'); raise SystemExit('accepted Feb 31')
except ValueError: pass
print('PASS')
