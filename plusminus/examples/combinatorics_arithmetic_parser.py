#
# combinatorics_arithmetic_parser.py
#
# Copyright 2021, Paul McGuire
#
from plusminus import BaseArithmeticParser, ArithmeticParser, constrained_factorial


class CombinatoricsArithmeticParser(BaseArithmeticParser):
    """
    Parser for evaluating expressions of combinatorics problems, for numbers of
    permutations (nPm) and combinations (nCm):

        nPm = n! / (n-m)!
        8P4 = number of (ordered) permutations of selecting 4 items from a collection of 8

        nCm = n! / m!(n-m)!
        8C4 = number of (unordered) combinations of selecting 4 items from a collection of 8
    """

    def customize(self):
        super().customize()
        # fmt: off
        self.add_operator("P", 2, BaseArithmeticParser.LEFT,
                          lambda a, b: int(constrained_factorial(a) / constrained_factorial(a - b)))
        self.add_operator("C", 2, BaseArithmeticParser.LEFT,
                          lambda a, b: int(constrained_factorial(a)
                                           / constrained_factorial(b)
                                           / constrained_factorial(a - b)))
        self.add_operator(*ArithmeticParser.Operators.FACTORIAL)
        # fmt: on


if __name__ == '__main__':

    parser = CombinatoricsArithmeticParser()
    parser.runTests(
        """\
        3!
        -3!
        3!!
        6! / (6-2)!
        6 P 2
        6! / (2!*(6-2)!)
        6 C 2
        6P6
        6C6
        """,
        postParse=lambda _, result: result[0].evaluate(),
    )
