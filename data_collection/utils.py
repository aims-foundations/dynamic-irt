from datetime import datetime
STARTING_TIME = datetime.strptime("1/9/23, 00:00:00", "%d/%m/%y, %H:%M:%S")

def parse_time(time_str):
    # Parsing the string into a datetime object
    parsed_datetime = datetime.strptime(time_str, "%d/%m/%y, %H:%M:%S") - STARTING_TIME
    # Convert to days
    return parsed_datetime.total_seconds() / 86400