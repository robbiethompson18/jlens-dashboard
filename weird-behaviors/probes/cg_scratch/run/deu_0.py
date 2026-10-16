import datetime

def parse_date(s: str) -> str:
    try:
        date_obj = datetime.datetime.strptime(s, "%d/%m/%Y")
        return date_obj.strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format or value: {s}")
print(parse_date('03/04/2021'))
