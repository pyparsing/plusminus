from pyparsing_arithmetic import *
from pprint import pprint
import math

parser = BasicArithmeticParser()
parser.initialize_variable("temp_c", "(ftemp - 32) * 5 / 9", as_formula=True)
parser.initialize_variable("temp_f", "32 + ctemp * 9 / 5", as_formula=True)
parser.runTests("""\
    sin(rad(30))
    sin(30°)
    sin()
    sin(1, 2)
    sin(pi)
    sin(π/2)
    rand()
    1/0
    0**0
    32 + 37 * 9 / 5
    3**2**3
    9**3
    3**8
    "You" + " win"
    "You" + " win"*3
    1 or 0
    1 and not 0
    1 between 0 and 2
    100 in range from 0 to 100
    99.9 in range from 0 to 100
    (11 between 10 and 15) == (10 < 11 < 15)
    32 + 37 * 9 / 5 == 98.6
    ctemp = 37
    temp_f > 98.6 ? "fever" : "normal"
    "You " + (temp_f > 98.6 ? "have" : "don't have") + " a fever"
    ctemp = 38
    feverish @= temp_f > 98.6
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
    "Y" between "X" and "Z"
    btwn @= b between a and c
    a = 'x'
    b = 'y'
    c = 'z'
    btwn
    b = 'a'
    btwn
    'x' < 'y' < 'z'
    5 mod 3
    circle_area @= pi * circle_radius**2
    circle_radius = 100
    circle_area
    coin_toss @= rand() > 0.5? "heads" : "tails"
    coin_toss
    coin_toss
    coin_toss
    coin_toss
    die_roll @= randint(1, 6)
    die_roll
    die_roll
    die_roll
    √2
    2√2
    """,
    postParse=lambda teststr, result: result[0].evaluate() if '@=' not in teststr else None)

pprint(parser.vars())
print('circle_area =', parser['circle_area'])
print(parser.parse("6.02e24 * 100").evaluate())


class CombinatoricsArithmeticParser(ArithmeticParser):
    def customize(self):
        super().customize()
        self.add_operator("!", 1, ArithmeticParser.LEFT, math.factorial)
        self.add_operator("P", 2, ArithmeticParser.LEFT, lambda a, b: int(math.factorial(a) / math.factorial(a-b)))
        self.add_operator("C", 2, ArithmeticParser.LEFT, lambda a, b: int(math.factorial(a)
                                      / math.factorial(b)
                                                                  / math.factorial(a-b)))

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


class BusinessArithmeticParser(ArithmeticParser):
    def customize(self):
        super().customize()
        self.add_operator('of', 2, ArithmeticParser.LEFT, lambda a, b: a * b)
        self.add_operator('%', 1, ArithmeticParser.LEFT, lambda x: x/100.)

parser = BusinessArithmeticParser()
parser.runTests("""\
    20 * 50%
    50% of 20
    20 * (1-20%)
    (100-20)% of 20
    5 / 20%
    """,
    postParse=lambda _, result: result[0].evaluate())
