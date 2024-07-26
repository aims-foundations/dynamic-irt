def parse_score(name):
    return float(name.split("\n")[1][1:])


def check(data):
    data = data["attemps"]
    print(f"Checking: n={len(data)}, sample: {data[0]}")