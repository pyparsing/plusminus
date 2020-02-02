from plusminus import ArithmeticParser, BasicArithmeticParser


class DiceRollParser(ArithmeticParser):
    def customize(self):
        import random
        super().customize()
        self.add_operator('d', 1, ArithmeticParser.RIGHT,
                                lambda x: random.randint(1, x))
        self.add_operator('d', 2, ArithmeticParser.LEFT,
                                lambda x, y: x * random.randint(1, y))

parser = DiceRollParser()
parser.runTests(['d20', '3d6', 'd20+3d4', 'd100'],
                postParse=lambda _, result: result[0].evaluate())



class DateTimeArithmeticParser(BasicArithmeticParser):
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

parser = DateTimeArithmeticParser()
parser.runTests("""\
    now()
    str(now())
    str(today())
    "A day from now: " + str(now() + 1d)
    "A day and an hour from now: " + str(now() + 1d + 1h)
    str(now() + 3*(1d + 1h))
    """, postParse=lambda _, result: result[0].evaluate())



class CombinatoricsParser(BasicArithmeticParser):
    def customize(self):
        import math
        super().customize()
        self.add_operator("P", 2, ArithmeticParser.LEFT, lambda a, b: int(math.factorial(a)
                                                                          / math.factorial(a - b)))
        self.add_operator("C", 2, ArithmeticParser.LEFT, lambda a, b: int(math.factorial(a)
                                                                          / math.factorial(b)
                                                                          / math.factorial(a - b)))
parser = CombinatoricsParser()
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



class BusinessArithmeticParser(ArithmeticParser):
    def customize(self):
        def pv(fv, rate, n_periods):
            return fv / (1 + rate) ** n_periods

        def fv(pv, rate, n_periods):
            return pv * (1 + rate) ** n_periods

        def pp(pv, rate, n_periods):
            return rate * pv / (1 - (1 + rate) ** (-n_periods))

        super().customize()
        self.add_operator("of", 2, ArithmeticParser.LEFT, lambda a, b: a * b)
        self.add_operator('%', 1, ArithmeticParser.LEFT, lambda x: x/100)
        self.add_function('PV', 3, pv)
        self.add_function('FV', 3, fv)
        self.add_function('PP', 3, pp)

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
