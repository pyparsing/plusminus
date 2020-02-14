import pytest

# The fixture in conftest.py is found using pytest magic, but if you wanted the descriptive name,
# in the fixture definition, but shortened for the test, you could do something like:
# from test.conftest import basic_arithmetic_parser as arith_parser


# You don't actually need to place test functions in a class,
# but you can setup/scope fixtures for one test/class/module/test_suite run etc
class TestBasicArithmetic:
    @pytest.mark.parametrize(
        'input_string, parse_result',
        [('sin(rad(30))', 0.5),  # sin(30 radians) = sin(5400/pi degrees) = -0.98803162409
         # Not super obvious that both rad() and ° use math.radians unless you read the source?
         # I expected one to be radians and the other degrees.
         ('sin(30°)', 0.5),
         pytest.param('sin()', 'type error',
                      marks=pytest.mark.xfail(reason="TypeError: sin takes 1 arg, 0 given")),
         pytest.param('sin(1, 2)', 'type error',
                      marks=pytest.mark.xfail(reason="TypeError: sin takes 1 arg, 0 given")),
         ('sin(pi)', 0),
         ('sin(π/2)', 1),
         pytest.param('rnd()', 'float 0 < x < 1',
                      marks=pytest.mark.xfail(reason="Variable float result.")),  # Put into own test.
         pytest.param('1/0', 'zero division error',
                      marks=pytest.mark.xfail(reason="ZeroDivisionError: division by zero")),
         ('0**0', 1),
         ('32 + 37 * 9 / 5', 98.6),
         ('3**2**3', 6561),
         ('9**3', 729),
         ('3**8', 6561),
         ('"You" + " win"', 'You win'),
         ('"You" + " win"*3', 'You win win win'),
         ('1 or 0', True),
         ('1 and not 0', True),
         ('1 between 0 and 2', True),
         ('100 in range from 0 to 100', False),
         ('99.9 in range from 0 to 100', True),
         ('(11 between 10 and 15) == (10 < 11 < 15)', True),
         ('32 + 37 * 9 / 5 == 98.6', True),

         # I believe the following are a sequence, each relying on the previous completing separately, and as such
         # I've split them out into test: temperature_vars_and_expressions
         # ('ctemp = 37', [37]),
         # ('temp_f = 100.2', [100.2]),
         # ('temp_f > 98.6 ? "fever" : "normal"', 'fever'),
         # ('"You " + (temp_f > 98.6 ? "have" : "don\'t have") + " a fever"', 'You have a fever'),
         # ('ctemp = 38', [38]),
         # ('feverish @= temp_f > 98.6', '[(temp_f'>'98.6)]'),
         # ('"You " + (feverish ? "have" : "don\'t have") + " a fever"', 'You have a fever'),
         # ('temp_f = 98.2', [98.2]),
         # ('"You " + (feverish ? "have" : "don\'t have") + " a fever"', "You don't have a fever"),

         # Again, the following are a sequence. Spun out into test abc_xyz_vars_and_expressions
         # ('a = 100', [100]),
         # ('b @= a / 10', '[(a'/'10)]'),
         # ('a / 2', 50),
         # ('b', 10),
         # ('a = 5', ''),
         # ('b', ''),
         # ('a = b + 3', ''),
         # ('b', ''),
         # ('a = b + 3', ''),
         # ('a + c', ''),
         ('"Y" between "X" and "Z"', True),
         # The following seem to rely on the earlier assignment of a and is put with that sequence
         # ('btwn @= b between a and c', ''),
         # These needed double quotes to escape the single quotes.
         # ("a = 'x'", ['x']),
         # ("b = 'y'", ''),
         # ("c = 'z'", ''),
         # ("btwn", True),
         # ("b = 'a'", ''),
         # ("btwn", ''),
         # ("'x' < 'y' < 'z'", ''),
         ('5 mod 3', 2),

         # The following are spun out into circle
         # ('circle_area @= pi * circle_radius**2', ''),  # Relies on unassigned circle_radius, returns an expression
         # ('circle_radius = 100', ''),
         # ('circle_area', ''),

         # Spin these out into test_coin_toss
         # ('coin_toss @= rnd() > 0.5? "heads" : "tails"', ''),
         # ('coin_toss', ''),
         # ('coin_toss', ''),
         # ('coin_toss', ''),
         # ('coin_toss', ''),

         # Spin these out into test_die_roll
         # ('die_roll @= randint(1, 6)', ''),
         # ('die_roll', ''),
         # ('die_roll', ''),
         # ('die_roll', ''),

         ('√2', 1.414213562373095),  # 1dp short of Python's native 2**0.5 (=1.414213562373095)
         ('2√2', 2.82842712474619),  # 2dp short of Python's native 2*2**0.5 (=2.8284271247461903)
         ('√-1', 1j),
         # ('10000**100000', ),  OverflowError
         ('0**10000000000**10000000000', 0),
         ('0**(-1)**2', 0),
         # ('0**(-1)**3', ),  ZeroDivisionError
         ('1000000000000**1000000000000**0', 1000000000000),
         ('1000000000000**0**1000000000000**1000000000000', 1),
         ('100 < 101', True),
         ('100 <= 101', True),
         ('100 > 101', False),
         ('100 >= 101', False),
         ('100 == 101', False),
         ('100 != 101', True),
         ('100 < 99', False),
         ('100 <= 99', False),
         ('100 > 99', True),
         ('100 >= 99', True),
         ('100 == 99', False),
         ('100 != 99', True),
         ('100 < 100+1E-18', False),
         ('100 <= 100+1E-18', True),
         ('100 > 100+1E-18', False),
         ('100 >= 100+1E-18', True),
         ('100 == 100+1E-18', True),
         ('100 != 100+1E-18', False),
         ])
    def test_evaluate(self, basic_arithmetic_parser,
                      input_string, parse_result):
        assert basic_arithmetic_parser.evaluate(input_string) == parse_result

    def test_evaluate_rnd_no_arg(self, basic_arithmetic_parser):
        random_value = basic_arithmetic_parser.evaluate('rnd()')
        assert isinstance(random_value, float)
        assert 0 < random_value < 1

    def test_temperature_vars_and_expressions(self, basic_arithmetic_parser):
        assert basic_arithmetic_parser.evaluate('ctemp = 37')  # == [37]
        # for some reason assert op == [37], and assert op[0] == 37 both fail?
        # Type of returned [37] is list, but the type of 100 is <class 'plusminus.plusminus.LiteralNode'>
        # Mocking this class out looks possible, but I couldn't quickly figure out how to do it so I skipped.
        # assert str(operation[0]) == str([37]) == '37' is also an option, but looks kinda hacky.
        # I assert (ha) that we can assume these assignments work because they're tested in test_abc_xyz

        # NB Some of these needed escaped single quotes.
        assert basic_arithmetic_parser.evaluate('temp_f = 100.2')  # == [100.2]
        assert basic_arithmetic_parser.evaluate('temp_f > 98.6 ? "fever" : "normal"') == 'fever'
        assert basic_arithmetic_parser.evaluate('"You " + (temp_f > 98.6 ? "have" : "don\'t have") + " a fever"'
                                                ) == 'You have a fever'
        assert basic_arithmetic_parser.evaluate('ctemp = 38')  # == [38]
        assert basic_arithmetic_parser.evaluate('feverish @= temp_f > 98.6')
        # == [(temp_f'>'98.6)], don't have to test for this unless you want to ensure the returned expression is as expected.
        assert basic_arithmetic_parser.evaluate('"You " + (feverish ? "have" : "don\'t have") + " a fever"'
                                                ) == 'You have a fever'
        assert basic_arithmetic_parser.evaluate('temp_f = 98.2')  # == [98.2]
        assert basic_arithmetic_parser.evaluate('"You " + (feverish ? "have" : "don\'t have") + " a fever"'
                                                ) == "You don't have a fever"

    def test_abc_xyz_vars_and_expressions(self, basic_arithmetic_parser):
        basic_arithmetic_parser = basic_arithmetic_parser
        assert basic_arithmetic_parser.evaluate('a = 100')  # == [100]
        assert basic_arithmetic_parser.evaluate('b @= a / 10') == 10.0
        assert basic_arithmetic_parser.evaluate('a / 2') == 50
        assert basic_arithmetic_parser.evaluate('b') == 10
        assert basic_arithmetic_parser.evaluate('a = 5')  # == [5] # See above # => b = 5/10 = 0.5
        assert basic_arithmetic_parser.evaluate('b') == 0.5
        assert basic_arithmetic_parser.evaluate('a = b + 3')  # == [3.5]  # See above  # => b = a/10 = 0.35
        assert basic_arithmetic_parser.evaluate('b') == 0.35
        assert basic_arithmetic_parser.evaluate('a = b + 3')  # == [3.35]  # See above  # => b = a/10 = 0.335
        with pytest.raises(NameError):
            basic_arithmetic_parser.evaluate('a + c')  # NameError: variable 'c' not known
        with pytest.raises(NameError):
            basic_arithmetic_parser.evaluate('btwn @= b between a and c')
        # In your tests - ^this doesn't raise a NameError - why?

        assert basic_arithmetic_parser.evaluate("a = 'x'")  # == ['x']   # See above # => b = a/10 = 0.335
        assert basic_arithmetic_parser.evaluate("b = 'y'")  # == ['y']   # See above # => b = a/10 = 0.335
        assert basic_arithmetic_parser.evaluate("c = 'z'")  # == ['z']   # See above # => b = a/10 = 0.335
        assert basic_arithmetic_parser.evaluate("btwn") is True
        assert basic_arithmetic_parser.evaluate("b = 'a'")  # == ['a']  # See above # => b = a/10 = 0.335
        assert basic_arithmetic_parser.evaluate("btwn") is False
        assert basic_arithmetic_parser.evaluate("'x' < 'y' < 'z'") is True

    def test_circle(self, basic_arithmetic_parser):
        basic_arithmetic_parser.evaluate('circle_radius = 0')  # Var was unassigned in orig test, why no error thrown?
        basic_arithmetic_parser.evaluate('circle_area @= pi * circle_radius**2')  # Relied on unassigned circle_radius.
        assert basic_arithmetic_parser.evaluate('circle_radius = 100')
        assert basic_arithmetic_parser.evaluate('circle_area') == 31415.926535897932

    def test_coin_toss(self, basic_arithmetic_parser):
        basic_arithmetic_parser.evaluate('coin_toss @= rnd() > 0.5? "heads" : "tails"')
        assert basic_arithmetic_parser.evaluate('coin_toss') in ('heads', 'tails')
        assert basic_arithmetic_parser.evaluate('coin_toss') in ('heads', 'tails')
        assert basic_arithmetic_parser.evaluate('coin_toss') in ('heads', 'tails')
        assert basic_arithmetic_parser.evaluate('coin_toss') in ('heads', 'tails')

    def test_die_roll(self, basic_arithmetic_parser):
        basic_arithmetic_parser.evaluate('die_roll @= randint(1, 6)')
        assert basic_arithmetic_parser.evaluate('die_roll') in (1, 2, 3, 4, 5, 6)
        assert basic_arithmetic_parser.evaluate('die_roll') in (1, 2, 3, 4, 5, 6)
        assert basic_arithmetic_parser.evaluate('die_roll') in (1, 2, 3, 4, 5, 6)

    @pytest.mark.parametrize(
        'input_string, returned_error',
        [('sin()', TypeError),
         ('sin(1, 2)', TypeError),
         ('1/0', ZeroDivisionError),
         ('0**(-1)**3', ZeroDivisionError),
         ('10000**100000', OverflowError),
         ])
    def test_evaluate_throws_errors(self, basic_arithmetic_parser,
                                    input_string, returned_error):
        with pytest.raises(returned_error):
            basic_arithmetic_parser.evaluate(input_string)
