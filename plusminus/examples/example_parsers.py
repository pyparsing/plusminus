#
# example_parsers.py
#
# Example domain-specific parsers extending the plusminus base classes
#
# Copyright 2020, Paul McGuire
#
from plusminus import ArithmeticParser, BasicArithmeticParser, safe_pow, constrained_factorial

__all__ = "DiceRollParser DateTimeArithmeticParser CombinatoricsArithmeticParser BusinessArithmeticParser".split()


class DiceRollParser(ArithmeticParser):
    """
    Parser for evaluating expressions representing rolls of dice, as used in many board and
    role-playing games, such as:

        d20
        3d20
        5d6 + d20
    """
    def customize(self):
        import random
        super().customize()
        self.add_operator('d', 1, ArithmeticParser.RIGHT,
                                lambda a: random.randint(1, a))
        self.add_operator('d', 2, ArithmeticParser.LEFT,
                                lambda a, b: sum(random.randint(1, b) for _ in range(a)))


class DateTimeArithmeticParser(ArithmeticParser):
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
        self.add_operator('d', 1, ArithmeticParser.LEFT, lambda t: t * DateTimeArithmeticParser.SECONDS_PER_DAY)
        self.add_operator('h', 1, ArithmeticParser.LEFT, lambda t: t * DateTimeArithmeticParser.SECONDS_PER_HOUR)
        self.add_operator('m', 1, ArithmeticParser.LEFT, lambda t: t * DateTimeArithmeticParser.SECONDS_PER_MINUTE)
        self.add_operator('s', 1, ArithmeticParser.LEFT, lambda t: t)
        self.add_function('now', 0, lambda: datetime.utcnow().timestamp())
        self.add_function('today', 0, lambda: datetime.utcnow().replace(hour=0,
                                                                        minute=0,
                                                                        second=0,
                                                                        microsecond=0).timestamp())
        self.add_function('str', 1, lambda dt: str(datetime.fromtimestamp(dt)))


class CombinatoricsArithmeticParser(BasicArithmeticParser):
    """
    Parser for evaluating expressions of combinatorics problems, for numbers of
    permutations (nPm) and combinations (nCm):

        nPm = n! / (n-m)!
        8P4 = number of (ordered) permutations of selecting 4 items from a collection of 8

        nCm = n! / m!(n-m)!
        8C4 = number of (unordered) combinations of selecting 4 items from a collection of 8
    """
    def customize(self):
        import math
        super().customize()
        self.add_operator("P", 2, ArithmeticParser.LEFT, lambda a, b: int(constrained_factorial(a)
                                                                          / constrained_factorial(a - b)))
        self.add_operator("C", 2, ArithmeticParser.LEFT, lambda a, b: int(constrained_factorial(a)
                                                                          / constrained_factorial(b)
                                                                          / constrained_factorial(a - b)))


class BusinessArithmeticParser(ArithmeticParser):
    """
    A parser for evaluating common financial and retail calculations:

        50% of 20
        20 * (1-20%)
        (100-20)% of 20
        5 / 20%
        FV(20000, 3%, 30)
        FV(20000, 3%/12, 30*12)

    Functions:
        FV(present_value, rate_per_period, number_of_periods)
            future value of an amount, n periods in the future, at an interest rate per period
        PV(future_value, rate_per_period, number_of_periods)
            present value of a future amount, n periods in the future, at an interest rate per period
        PP(present_value, rate_per_period, number_of_periods)
            periodic value of n amounts, one per period, for n periods, at an interest rate per period
    """
    def customize(self):
        def pv(fv, rate, n_periods):
            return fv / safe_pow(1 + rate, n_periods)

        def fv(pv, rate, n_periods):
            return pv * safe_pow(1 + rate, n_periods)

        def pp(pv, rate, n_periods):
            return rate * pv / (1 - safe_pow(1 + rate, -n_periods))

        super().customize()
        self.add_operator("of", 2, ArithmeticParser.LEFT, lambda a, b: a * b)
        self.add_operator('%', 1, ArithmeticParser.LEFT, lambda a: a / 100)
        self.add_function('PV', 3, pv)
        self.add_function('FV', 3, fv)
        self.add_function('PP', 3, pp)


if __name__ == '__main__':

    parser = DiceRollParser()
    parser.runTests(['d20', '3d6', 'd20+3d4', '2d100'],
                    postParse=lambda _, result: result[0].evaluate())

    parser = DateTimeArithmeticParser()
    parser.runTests("""\
        now()
        str(now())
        str(today())
        "A day from now: " + str(now() + 1d)
        "A day and an hour from now: " + str(now() + 1d + 1h)
        str(now() + 3*(1d + 1h))
        """, postParse=lambda _, result: result[0].evaluate())

    parser = CombinatoricsArithmeticParser()
    parser.runTests("""\
        3!
        -3!
        3!!
        6! / (6-2)!
        6 P 2
        6! / (2!*(6-2)!)
        6 C 2
        6P6
        6C6
        """, postParse=lambda _, result: result[0].evaluate())

    parser = BusinessArithmeticParser()
    parser.runTests("""\
        25%
        20 * 50%
        50% of 20
        20 * (1-20%)
        (100-20)% of 20
        5 / 20%
        FV(20000, 3%, 30)
        FV(20000, 3%/12, 30*12)
        """, postParse=lambda _, result: result[0].evaluate())
