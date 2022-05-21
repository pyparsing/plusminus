#
# dice_roll_parser.py
#
# Copyright 2021, Paul McGuire
#
from plusminus import BaseArithmeticParser


# fmt: off
class DiceRollParser(BaseArithmeticParser):
    """
    Parser for evaluating expressions representing rolls of dice, as used in many board and
    role-playing games, such as:

        d20
        3d20
        5d6 + d20
        min(d6, d6, d6)
        show(d6, d6, d6)
    """

    def customize(self):
        import random

        self.add_operator("d", 1, BaseArithmeticParser.RIGHT, lambda a: random.randint(1, a))
        self.add_operator("d", 2, BaseArithmeticParser.LEFT,
                          lambda a, b: sum(random.randint(1, b) for _ in range(a)))

        self.add_function("min", ..., min)
        self.add_function("max", ..., max)
        self.add_function("show", ..., lambda *args: (list(args), sum(args)))

# fmt: on


if __name__ == '__main__':

    parser = DiceRollParser()
    parser.runTests(
        """\
        d20
        3d6
        d20+3d4
        2d100
        max(d6, d6, d6)
        show(d6, d6, d6)        
        """,
        postParse=lambda _, result: result[0].evaluate(),
    )
