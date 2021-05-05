#
# date_time_arithmetic_parser.py
#
# Copyright 2021, Paul McGuire
#
from plusminus import BaseArithmeticParser


class DateTimeArithmeticParser(BaseArithmeticParser):
    """
    Parser for evaluating expressions in dates and times, using operators d, h, m, and s
    to define terms for amounts of days, hours, minutes, and seconds:

        now()
        today()
        now() + 10s
        now() + 24h

    All numeric expressions will be treated as UTC integer timestamps. To display
    timestamps as ISO strings, use str():

        str(now())
        str(today() + 3d)
    """

    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60
    SECONDS_PER_DAY = SECONDS_PER_HOUR * 24

    def customize(self):
        from datetime import datetime

        # fmt: off
        self.add_operator("d", 1, BaseArithmeticParser.LEFT, lambda t: t * DateTimeArithmeticParser.SECONDS_PER_DAY)
        self.add_operator("h", 1, BaseArithmeticParser.LEFT, lambda t: t * DateTimeArithmeticParser.SECONDS_PER_HOUR)
        self.add_operator("m", 1, BaseArithmeticParser.LEFT, lambda t: t * DateTimeArithmeticParser.SECONDS_PER_MINUTE)
        self.add_operator("s", 1, BaseArithmeticParser.LEFT, lambda t: t)

        self.add_function("now", 0, lambda: datetime.utcnow().timestamp())
        self.add_function("today", 0,
                          lambda: datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        self.add_function("str", 1, lambda dt: str(datetime.fromtimestamp(dt)))
        # fmt: on


if __name__ == "__main__":

    parser = DateTimeArithmeticParser()
    parser.runTests(
        """\
        now()
        str(now())
        str(today())
        "A day from now: " + str(now() + 1d)
        "A day and an hour from now: " + str(now() + 1d + 1h)
        str(now() + 3*(1d + 1h))
        """,
        postParse=lambda _, result: result[0].evaluate(),
    )
