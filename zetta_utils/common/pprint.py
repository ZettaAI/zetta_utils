import datetime


def lrpad(
    string: str = "", level: int = 1, length: int = 80, bounds: str = "|", filler: str = " "
) -> str:  # pragma: no cover
    newstr = ""
    newstr += bounds
    while len(newstr) < level * 4:
        newstr += filler
    newstr += string
    if len(newstr) >= length:
        return newstr
    while len(newstr) < length - 1:
        newstr += filler
    return newstr + bounds


def lrpadprint(
    string: str = "", level: int = 1, length: int = 80, bounds: str = "|", filler: str = " "
) -> None:  # pragma: no cover
    print(lrpad(string=string, level=level, length=length, bounds=bounds, filler=filler))


def utcnow_ISO8601() -> str:  # pragma: no cover # pylint: disable=invalid-name
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
