#
# arith_tests.py
#
# Demo/tests of various plusminus parsers
#
# Copyright 2020, Paul McGuire
#
from plusminus import *
from plusminus.examples.example_parsers import DiceRollParser, CombinatoricsArithmeticParser, BusinessArithmeticParser
from pprint import pprint

import sys
sys.setrecursionlimit(2000)

parser = BasicArithmeticParser()
parser.initialize_variable("temp_c", "(ftemp - 32) * 5 / 9", as_formula=True)
parser.initialize_variable("temp_f", "32 + ctemp * 9 / 5", as_formula=True)
parser.runTests("""\
    sin(rad(30))
    sin(30°)

    # test rejection of functions with wrong number of args
    sin()
    sin(1, 2)
    hypot(1)

    sin(pi)
    sin(π/2)
    rnd()
    1/0
    0**0
    32 + 37 * 9 / 5
    
    # verify right-to-left eval of exponents
    # 3**2**3 should eval as 3**(2**3)
    3**2**3
    3**(2**3)
    (3**2)**3

    # addition and multiplication of strings
    "You" + " win"
    "You" + " win"*3

    # ints as bools
    not not 0
    not not 1
    1 or 0
    1 and not 0

    # in operator
    1 in (0, 2)
    100 in [0, 100)
    99.9999 in [0, 100)
    (11 in (10,15)) == (10 < 11 < 15)
    2 in {1, 2, 3}

    32 + 37 * 9 / 5 == 98.6
    ctemp = 37
    temp_f = 100.2
    temp_f > 98.6 ? "fever" : "normal"
    "You " + (temp_f > 98.6 ? "have" : "don't have") + " a fever"
    ctemp = 38
    feverish @= temp_f > 98.6
    "You " + (feverish ? "have" : "don't have") + " a fever"    
    temp_f = 98.2
    "You " + (feverish ? "have" : "don't have") + " a fever"    
    a = 100
    b @= a / 10
    a / 2
    b
    a = 5
    b
    a = b + 3
    b
    a = b + 3
    a + c
    
    'x' < 'y' < 'z'
    5 mod 3
    circle_area @= pi * circle_radius**2
    circle_radius = 100
    circle_area
    coin_toss @= rnd() > 0.5? "heads" : "tails"
    coin_toss
    coin_toss
    coin_toss
    coin_toss
    die_roll @= randint(1, 6)
    die_roll
    die_roll
    die_roll
    
    # unary and binary square root
    # and imaginary square root
    √2
    2√2
    √-1

    # test safe_pow
    10000**100000
    0**10000000000**10000000000
    0**(-1)**2
    0**(-1)**3
    1000000000000**1000000000000**0
    1000000000000**0**1000000000000**1000000000000
    
    # test all comparison operators
    100 < 101
    100 <= 101
    100 > 101
    100 >= 101
    100 == 101
    100 != 101
    100 < 99
    100 <= 99
    100 > 99
    100 >= 99
    100 == 99
    100 != 99
    100 < 100+1E-18
    100 <= 100+1E-18
    100 > 100+1E-18
    100 >= 100+1E-18
    100 == 100+1E-18
    100 != 100+1E-18
    
    # range checks
    100 in [100, 101]
    100 in [100, 101)
    100 in (100, 101]
    100 in (100, 101)
    100.5 in (100, 101)
    
    # in range with strings
    "Y" in ("X", "Z")
    btwn @= b in (a,c)
    a = 'x'
    b = 'y'
    c = 'z'
    btwn
    b = 'a'
    btwn

    # function call with variable number of args
    hypot(3, 4)
    nhypot(3, 4)
    nhypot(3, 4) == hypot(3, 4)
    nhypot(3, 4, 5, 6) == hypot(3, hypot(4, hypot(5, 6)))
    nhypot()
    
    # set operations
    a, b = 1, 10
    1 in (a, b)
    1 in [a, b)
    1 not in [a, b)
    1 ∈ [a, b)
    1 ∉ [a, b)
    1 in { a, 11, 22, 53}
    1 not in {b, 0}
    myset = { a, 11, 22, 53}
    myset
    1 in myset
    { 0, 2, 22}
    { a, 11, 22, 53} ∩ { 0, 2, 22}
    { a, 11, 22, 53} ∪ { 0, 2, 22}
    { a, 11, 22, 53} ∩ {}
    { a, 11, 22, 53} ∪ {}
    myset ∩ { 0, 2, 22}
    myset ∪ { 0, 2, 22}
    1 in (myset ∩ { 0, 2, 22})
    1 in (myset ∪ { 0, 2, 22})
    1 ∈ (myset ∩ { 0, 2, 22})
    1 ∉ (myset ∪ { 0, 2, 22})
    1 in (myset ∩ {})
    1 in (myset ∪ {})
    
    # mismatched parentheses
    5 + (3*
    """,
    postParse=lambda teststr, result: result[0].evaluate() if '@=' not in teststr else None)

pprint(parser.vars())
print('circle_area =', parser['circle_area'])
print(parser.parse("6.02e24 * 100").evaluate())


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
    """,
    postParse=lambda _, result: result[0].evaluate())


parser = BusinessArithmeticParser()
parser.runTests("""\
    25%
    20 * 50%
    50% of 20
    20 * (1-20%)
    (100-20)% of 20
    5 / 20%
    FV(20000, 3%, 30)
    PV(FV(20000, 3%, 30), 3%, 30)
    FV(20000, 3%/12, 30*12)
    """,
    postParse=lambda _, result: result[0].evaluate())

from datetime import datetime
class DateTimeArithmeticParser(ArithmeticParser):
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60
    SECONDS_PER_DAY = SECONDS_PER_HOUR * 24
    def customize(self):
        super().customize()
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

parser = DateTimeArithmeticParser()
parser.runTests("""\
    now()
    str(now())
    str(today())
    "A day from now: " + str(now() + 1d)
    "A day and an hour from now: " + str(now() + 1d + 1h)
    str(now() + 3*(1d + 1h))
    """,
    postParse=lambda _, result: result[0].evaluate())


parser = DiceRollParser()
parser.runTests("""
d20
3d6
d20 + 3d4
(3d6)/3
""", postParse=lambda _, result: result[0].evaluate())


print()


# override max number of variables
class restore:
    """
    Context manager for restoring an object's attributes back the way they were if they were
    changed or deleted, or to remove any attributes that were added.
    """
    def __init__(self, obj, *attr_names):
        self._obj = obj
        self._attrs = attr_names
        if not self._attrs:
            self._attrs = [name for name in vars(obj) if name not in ('__dict__', '__slots__')]
        self._no_attr_value = object()
        self._save_values = {}

    def __enter__(self):
        for attr in self._attrs:
            self._save_values[attr] = getattr(self._obj, attr, self._no_attr_value)
        return self

    def __exit__(self, *args):
        for attr in self._attrs:
            save_value = self._save_values[attr]
            if save_value is not self._no_attr_value:
                if getattr(self._obj, attr, self._no_attr_value) != save_value:
                    print("reset", attr, "to", save_value)
                    setattr(self._obj, attr, save_value)
            else:
                if hasattr(self._obj, attr):
                    delattr(self._obj, attr)


print('test defining too many vars (set max to 20)')
with restore(ArithmeticParser):
    ArithmeticParser.MAX_VARS = 20
    parser = ArithmeticParser()
    try:
        for i in range(1000):
            parser.evaluate("a{} = 0".format(i))
    except Exception as e:
        print(len(parser.vars()), ArithmeticParser.MAX_VARS)
        print("{}: {}".format(type(e).__name__, e))
