#
# business_arithmetic_parser.py
#
# Copyright 2021, Paul McGuire
#
from plusminus import BaseArithmeticParser, safe_pow


class BusinessArithmeticParser(BaseArithmeticParser):
    """
    A parser for evaluating common financial and retail calculations:

        50% of 20
        20 * (1-20%)
        (100-20)% of 20
        20% off of 20
        20 less 20%
        5 / 20%
        FV(20000, 3%, 30)
        FV(20000, 3%/12, 30*12)

    Operators:
        % - convert number to a percentage ("20%" -> 0.2)
        of - synonym for multiplication, used in "X% of Y" expressions ("20% of 5" -> 1)
        off - discount, computed as (1-x), used in "X% off of Y" expressions ("20% off of 5" -> 4)
        less - discount, used in "Y less X%" expressions ("5 less 20%" -> 4)

    Functions:
        FV(present_value, rate_per_period, number_of_periods)
            future value of an amount, n periods in the future, at an interest rate per period
        PV(future_value, rate_per_period, number_of_periods)
            present value of a future amount, n periods in the future, at an interest rate per period
        PP(present_value, rate_per_period, number_of_periods)
            periodic value of n amounts, one per period, for n periods, at an interest rate per period
    """

    def customize(self):
        def pv(future_value, rate, n_periods):
            return future_value / safe_pow(1 + rate, n_periods)

        def fv(present_value, rate, n_periods):
            return present_value * safe_pow(1 + rate, n_periods)

        def pp(present_value, rate, n_periods):
            return rate * present_value / (1 - safe_pow(1 + rate, -n_periods))

        super().customize()
        self.add_operator("of", 2, BaseArithmeticParser.LEFT, lambda a, b: a * b)
        self.add_operator("less", 2, BaseArithmeticParser.LEFT, lambda a, b: (1-b) * a)
        self.add_operator("off", 1, BaseArithmeticParser.LEFT, lambda a: 1-a)
        self.add_operator("%", 1, BaseArithmeticParser.LEFT, lambda a: a / 100)

        self.add_function("PV", 3, pv)
        self.add_function("FV", 3, fv)
        self.add_function("PP", 3, pp)


if __name__ == '__main__':

    parser = BusinessArithmeticParser()
    parser.runTests(
        """\
        25%
        20 * 50%
        50% of 20
        (100-30)% of 50
        30% off
        30% off of 50
        50 less 30%
        20 * (1-20%)
        (100-20)% of 20
        5 / 20%
        FV(20000, 3%, 30)
        FV(20000, 3%/12, 30*12)
        """,
        postParse=lambda _, result: result[0].evaluate(),
    )
