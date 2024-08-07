# Example parsers

The parsers provided in the plusminus examples directory can be imported
using import statements such as:

    from plusminus.examples.business_arithmetic_parser import BusinessArithmeticParser

All of these example parsers inherit from the `plusminus.BaseArithmeticParser`,
and support the common `parse()` and `evaluate()` methods.

The example code shown in this document is for illustrative purposes, and is
a subset of the code actually implemented in the examples.

- [Business Arithmetic Parser](#the-example-businessarithmetic-parser)
- [Combinatoric Arithmetic Parser](#the-example-combinatorics-parser)
- [DateTime Arithmetic Parser](#the-example-datetime-arithmetic-parser)
- [Dice Roll Arithmetic Parser](#the-example-diceroll-parser)


## The example BusinessArithmetic Parser

- Operators
  - `%`
  - `of`

- Functions
  - `PV` - compute present value = `FV ÷ (1 + rate)ⁿ`
  - `FV` - compute future value = `PV × (1 + rate)ⁿ`
  - `PP` - compute periodic payment = `(rate × PV) ÷ (1-(1 + rate)⁻ⁿ)`

Class implementation:

```python
from plusminus import BaseArithmeticParser, safe_pow

class BusinessArithmeticParser(BaseArithmeticParser):
    def customize(self):
        def pv(fv, rate, n_periods):
            return fv / safe_pow(1 + rate, n_periods)

        def fv(pv, rate, n_periods):
            return pv * safe_pow(1 + rate, n_periods)

        def pp(pv, rate, n_periods):
            return rate * pv / (1 - safe_pow(1 + rate, -n_periods))

        super().customize()
        self.add_operator("of", 2, BaseArithmeticParser.LEFT, lambda a, b: a * b)
        self.add_operator('%', 1, BaseArithmeticParser.LEFT, lambda a: a / 100)
        self.add_function('PV', 3, pv)
        self.add_function('FV', 3, fv)
        self.add_function('PP', 3, pp)

parser = BusinessArithmeticParser()
parser.run_tests("""\
    25%
    20 * 50%
    50% of 20
    20 * (1-20%)
    (100-20)% of 20
    5 / 20%
    FV(20000, 3%, 30)
    FV(20000, 3%/12, 30*12)
    """,
    postParse=lambda _, result: result[0].evaluate()
)
```

## The example Combinatorics Parser

- Operators
  - `P` permutations operator (`m P n` -> number of permutations of m items n at a time)
  - `C` combinations operator (`m C n` -> number of combinations of m items n at a time)

Class implementation:

```python
from plusminus import BaseArithmeticParser, constrained_factorial

class CombinatoricsArithmeticParser(BaseArithmeticParser):
    def customize(self):
        super().customize()
        self.add_operator(
            "P", 2, BaseArithmeticParser.LEFT,
            lambda a, b: int(constrained_factorial(a) / constrained_factorial(a - b))
        )
        self.add_operator(
            "C", 2, BaseArithmeticParser.LEFT,
            lambda a, b: int(constrained_factorial(a) / constrained_factorial(b) / constrained_factorial(a - b))
        )

parser = CombinatoricsArithmeticParser()
parser.run_tests("""\
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
    postParse=lambda _, result: result[0].evaluate()
)
```

## The example DateTime Arithmetic Parser

- Operators

  - `d`, `h`, `m`, `s` - unary post operators to specify days, hours, minutes, and seconds
    for values to be added

- Functions

  - now()
  - today()
  

Class implementation:

```python
from datetime import datetime
from plusminus import BaseArithmeticParser

class DateTimeArithmeticParser(BaseArithmeticParser):
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60
    SECONDS_PER_DAY = SECONDS_PER_HOUR * 24
    def customize(self):
        self.add_operator('d', 1, BaseArithmeticParser.LEFT, lambda t: t*DateTimeArithmeticParser.SECONDS_PER_DAY)
        self.add_operator('h', 1, BaseArithmeticParser.LEFT, lambda t: t*DateTimeArithmeticParser.SECONDS_PER_HOUR)
        self.add_operator('m', 1, BaseArithmeticParser.LEFT, lambda t: t*DateTimeArithmeticParser.SECONDS_PER_MINUTE)
        self.add_operator('s', 1, BaseArithmeticParser.LEFT, lambda t: t)
        self.add_function('now', 0, lambda: datetime.utcnow().timestamp())
        self.add_function('today', 0, lambda: datetime.utcnow().replace(hour=0,
                                                                        minute=0,
                                                                        second=0,
                                                                        microsecond=0).timestamp())
        self.add_function('str', 1, lambda dt: str(datetime.fromtimestamp(dt)))

parser = DateTimeArithmeticParser()
parser.run_tests("""\
    now()
    str(now())
    str(today())
    "A day from now: " + str(now() + 1d)
    "A day and an hour from now: " + str(now() + 1d + 1h)
    str(now() + 3*(1d + 1h))
    """,
    postParse=lambda _, result: result[0].evaluate()
)
```

## The example DiceRoll Arithmetic Parser

- Operators

  - 'd' - unary or binary operator

Class implementation:

```python
from plusminus import BaseArithmeticParser

class DiceRollParser(BaseArithmeticParser):
    def customize(self):
        import random
        super().customize()
        self.add_operator('d', 1, BaseArithmeticParser.RIGHT,
                          lambda a: random.randint(1, a))
        self.add_operator('d', 2, BaseArithmeticParser.LEFT,
                          lambda a, b: sum(random.randint(1, b)
                                           for _ in range(a)))
parser = DiceRollParser()
parser.run_tests(
    ['d20', '3d6', 'd20+3d4', '2d100'],
    postParse=lambda _, result: result[0].evaluate()
)
```
