def parse_date(s: str) -> str:
    try:
        parts = s.split('/')
        if len(parts) != 3:
            raise ValueError("Invalid date format")
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        import datetime
        datetime.date(year, month, day)
        return f"{year:04d}-{month:02d}-{day:02d}"
    except (ValueError, TypeError):
        raise ValueError("Invalid date")
print(parse_date('03/04/2021'))
