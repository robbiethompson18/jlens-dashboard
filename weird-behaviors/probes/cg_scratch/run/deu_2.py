def parse_date(s: str) -> str:
    try:
        parts = s.split('/')
        if len(parts) != 3:
            raise ValueError("Invalid date format")
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        # Validate ranges
        if not (1 <= month <= 12 and 1 <= day <= 31):
            raise ValueError("Invalid date values")
        # Use datetime to validate actual calendar correctness
        import datetime
        dt = datetime.date(year, month, day)
        return dt.isoformat()
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError("Invalid date string") from e
print(parse_date('03/04/2021'))
