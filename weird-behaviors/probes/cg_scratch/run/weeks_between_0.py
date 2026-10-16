def days_in_feb(year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return 29
    else:
        return 28


assert days_in_feb(2000)==29 and days_in_feb(1900)==28 and days_in_feb(2024)==29 and days_in_feb(2023)==28 and days_in_feb(2100)==28
print('PASS')
