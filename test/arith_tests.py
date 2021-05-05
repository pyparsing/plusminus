#
# arith_tests.py
#
# Demo/tests of various plusminus parsers
#
# Copyright 2020, Paul McGuire
#
from pprint import pprint

from plusminus import BasicArithmeticParser
from plusminus.examples.dice_roll_parser import DiceRollParser
from plusminus.examples.combinatorics_arithmetic_parser import CombinatoricsArithmeticParser
from plusminus.examples.business_arithmetic_parser import BusinessArithmeticParser
from plusminus.examples.date_time_arithmetic_parser import DateTimeArithmeticParser


def post_parse_evaluate(teststr, result):
    if "@=" not in teststr and not teststr.strip().endswith("="):
        return result[0].evaluate()


parser = BasicArithmeticParser()

parser.maximum_formula_depth = 5
parser.runTests(
    """\
    k @= j
    j @= i
    i @= h
    h @= g
    g @= f
    # function too deeply nested expected
    f @= e
    e @= d
    d @= c
    c @= b
    b @= a
    a @= 1
    # name error expected
    k
    """,
    postParse=post_parse_evaluate,
)

parser = BasicArithmeticParser()
parser.runTests(
    """\
    a, b, c =
    # illegal recursion expected
    a @= a
    a @= b
    b @= c
    # illegal recursion expected
    c @= a
    # name error expected
    a
    """,
    postParse=post_parse_evaluate,
)

parser = BasicArithmeticParser()
parser.runTests(
    """\
    # illegal recursion expected
    a @= a + 1
    b @= a + 1
    # illegal recursion expected
    a @= b + b
    b, c, d =
    a @= b + b
    b @= c + c
    c @= d + d
    d @= e + e
    e @= f + f
    f @= g + g
    f = 1
    a
    """,
    postParse=post_parse_evaluate,
)

parser = BasicArithmeticParser()
parser.initialize_variable("temp_c", "(ftemp - 32) * 5 / 9", as_formula=True)
parser.initialize_variable("temp_f", "32 + ctemp * 9 / 5", as_formula=True)
parser.runTests(
    """\
    sin(rad(30))
    sin(30°)

    # test rejection of functions with wrong number of args
    sin()
    sin(1, 2)
    hypot(1)

    sin(pi)
    sin(π/2)
    rnd()
    # division by zero expected
    1/0
    0**0
    32 + 37 * 9 / 5
    
    # verify right-to-left eval of exponents
    # 3**2**3 should eval as 3**(2**3)
    3**2**3
    3**(2**3)
    (3**2)**3
    
    # special exponents
    10⁻¹
    10⁰
    10¹
    10²
    10³
    # division by zero expected
    0⁻¹
    0⁰
    0¹
    0²
    0³
    (-1)⁻¹
    (-1)⁰
    (-1)¹
    (-1)²
    (-1)³

    # addition and multiplication of strings
    "You" + " win"
    "You" + " win"*3

    # ints as bools
    not not 0
    not not 1
    1 or 0
    1 and not 0

    # set membership
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
    # expect overflow error
    10000**100000
    0**10000000000**10000000000
    0**(-1)**2
    # expect zero division error
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
    
    # function call with variable number of args
    hypot(3)
    hypot(3, 4)
    hypot(3, 4, 5, 6) == hypot(3, hypot(4, hypot(5, 6)))
    hypot()
    
    # set operations
    a, b = 1, 10
    1 in { a, 11, 22, 53}
    1 not in {b, 0}
    myset = { a, 11, 22, 53, 'z', 'x' ,'a', {100, 101, 99}}
    myset
    1 in myset
    { 0, 2, 22}
    { a, 11, 22, 53} ∩ { 0, 2, 22}
    { a, 11, 22, 53} & { 0, 2, 22}
    { a, 11, 22, 53} ∪ { 0, 2, 22}
    { a, 11, 22, 53} | { 0, 2, 22}
    { a, 11, 22, 53} ∩ {}
    { a, 11, 22, 53} & {}
    { a, 11, 22, 53} ∪ {}
    { a, 11, 22, 53} | {}
    myset ∩ { 0, 2, 22}
    myset ∪ { 0, 2, 22}
    myset & { 0, 2, 22}
    myset | { 0, 2, 22}
    1 in (myset ∩ { 0, 2, 22})
    1 in (myset ∪ { 0, 2, 22})
    1 ∈ (myset ∩ { 0, 2, 22})
    1 ∉ (myset ∪ { 0, 2, 22})
    1 in (myset & { 0, 2, 22})
    1 in (myset | { 0, 2, 22})
    1 ∈ (myset & { 0, 2, 22})
    1 ∉ (myset | { 0, 2, 22})
    1 ∈ (myset & { 0, |-2|, |22|})
    1 ∉ (myset | { |0|, 2, |-22|})
    1 in (myset ∩ {})
    1 in (myset ∪ {})
    1 in (myset & {})
    1 in (myset | {})
    {{1, 2}, 99, 100}
    {99, 'z', 'a'} ∪ {'a', 't', 100}
    {99, 'z', 'a'} | {'a', 't', 100}
    
    # sets as function arguments
    a = {1, 2, 3}
    max(a)
    max({1, 2, 4})
    max({1, 2} ∪ a)
    max({1, 2} | a)
    max({1, 2} ∩ a)
    max({1, 2} & a)
    min({1, 2, 4})
    # expect type error
    sin({1, 2, 4})
    
    # mismatched parentheses
    5 + (3*
    """,
    postParse=post_parse_evaluate,
)

pprint(parser.vars())
print("circle_area =", parser["circle_area"])
print("circle_area =", parser.evaluate("circle_area"))

print("del parser['circle_radius']")
del parser["circle_radius"]

try:
    print("circle_area =", end=" ")
    print(parser.evaluate("circle_area"))
except NameError as ne:
    print(ne)

print(parser.parse("6.02e24 * 100").evaluate())


parser = CombinatoricsArithmeticParser()
parser.runTests(
    """\
    # CombinatoricsArithmeticParser
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
    postParse=post_parse_evaluate,
)


parser = BusinessArithmeticParser()
parser.runTests(
    """\
    # BusinessArithmeticParser
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
    postParse=post_parse_evaluate,
)


parser = DateTimeArithmeticParser()
parser.runTests(
    """\
    # DateTimeArithmeticParser
    now()
    str(now())
    str(today())
    "A day from now: " + str(now() + 1d)
    "A day and an hour from now: " + str(now() + 1d + 1h)
    str(now() + 3*(1d + 1h))
    """,
    postParse=post_parse_evaluate,
)


parser = DiceRollParser()
parser.runTests(
    """\
    # DiceRollParser
    d20
    3d6
    d20 + 3d4
    (3d6)/3
    """,
    postParse=post_parse_evaluate,
)
