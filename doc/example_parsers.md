## The example BusinessParser

- Operators
  - `%`
  - `of`

- Functions
  - `PV` - compute present value = `FV ÷ (1 + rate)ⁿ`
  - `FV` - compute future value = `PV × (1 + rate)ⁿ`
  - `PP` - compute periodic payment = `(rate × PV) ÷ (1-(1 + rate)⁻ⁿ)`


    class BusinessArithmeticParser(ArithmeticParser):
        def customize(self):
            def pv(fv, rate, n_periods):
                return fv / (1 + rate)**n_periods

            def fv(pv, rate, n_periods):
                return pv * (1 + rate)**n_periods

            def pp(pv, rate, n_periods):
                return rate * pv / (1 - (1 + rate)**(-n_periods))

            super().customize()
            self.add_operator('%', 1, ArithmeticParser.LEFT, math.radians)
            self.add_operator("of", 2, ArithmeticParser.LEFT, lambda a, b: a * b)
            self.add_function('PV', 3, pv)
            self.add_function('FV', 3, fv)
            self.add_function('PP', 3, pp)


## The example Combinatorics Parser

- Operators
  - `P` permutations operator (`m P n` -> number of permutations of m items n at a time)
  - `C` combinations operator (`m C n` -> number of combinations of m items n at a time)


    class CombinatoricsParser(BasicArithmeticParser):
        def customize(self):
            super().customize()
            self.add_operator("P", 2, ArithmeticParser.LEFT, lambda a, b: int(math.factorial(a)
                                                                              / math.factorial(a-b)))
            self.add_operator("C", 2, ArithmeticParser.LEFT, lambda a, b: int(math.factorial(a)
                                                                              / math.factorial(b)
                                                                              / math.factorial(a-b)))

## The example DateTimeParser

- Operators

  - `d`, `h`, `m`, `s` - unary post operators to specify days, hours, minutes, and seconds
    for values to be added

- Functions

  - now()
  - today()
  

    from datetime import datetime

    class DateTimeArithmeticParser(ArithmeticParser):
        SECONDS_PER_MINUTE = 60
        SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60
        SECONDS_PER_DAY = SECONDS_PER_HOUR * 24
        def customize(self):
            self.add_operator('d', 1, ArithmeticParser.LEFT, lambda t: t*DateTimeArithmeticParser.SECONDS_PER_DAY)
            self.add_operator('h', 1, ArithmeticParser.LEFT, lambda t: t*DateTimeArithmeticParser.SECONDS_PER_HOUR)
            self.add_operator('m', 1, ArithmeticParser.LEFT, lambda t: t*DateTimeArithmeticParser.SECONDS_PER_MINUTE)
            self.add_operator('s', 1, ArithmeticParser.LEFT, lambda t: t)
            self.add_function('now', 0, lambda: datetime.utcnow().timestamp())
            self.add_function('today', 0, lambda: datetime.utcnow().replace(hour=0,
                                                                            minute=0,
                                                                            second=0,
                                                                            microsecond=0).timestamp())
            self.add_function('str', 1, lambda dt: str(datetime.fromtimestamp(dt)))
