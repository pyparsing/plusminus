import pytest
from plusminus import BasicArithmeticParser, ArithmeticParseException
import sys

sys.setrecursionlimit(4000)


# Normally this would be kept in a "fixtures.py" file or similar, for access by all test scripts.
@pytest.fixture
def basic_arithmetic_parser():
    arith_parser = BasicArithmeticParser()

    # Add temp conversions - you could do this in the test, or create a separate/inheiting fixture with these.
    arith_parser.initialize_variable("temp_c", "(ftemp - 32) * 5 / 9", as_formula=True)
    arith_parser.initialize_variable("temp_f", "32 + ctemp * 9 / 5", as_formula=True)
    arith_parser.initialize_variable("ctemp", 38)

    return arith_parser


# You don't actually need to place test functions in a class,
# but you can setup/scope fixtures for one test/class/module/test_suite run etc
class TestBasicArithmetic:

    def _test_evaluate(self, basic_arithmetic_parser, input_string, expected_value):
        assert basic_arithmetic_parser.evaluate(input_string) == expected_value

    @pytest.mark.parametrize(
        "input_string, expected_value",
        [
            ("sin(rad(30))", 0.5),
            ("sin(30°)", 0.5),
            ("sin(π/2)", 1.0),
            ("0**0", 1),
            ("3**2**3", 3 ** 2 ** 3),
            ('"You" + " win"*3', "You win win win"),
            ("100 in [0, 100)", False),
            ("99.9 in [0, 100)", True),
            ("(0)", 0),
            ("((0))", 0),
            ("(((0)))", 0),
            ("((((0))))", 0),
            ("(((((0)))))", 0),
            ("((((((0))))))", 0),
            ("{{{{{{100}}}}}}",
             frozenset(
                 [
                     frozenset(
                         [
                             frozenset(
                                 [
                                     frozenset(
                                         [
                                             frozenset(
                                                 [
                                                     frozenset(
                                                         [
                                                             100
                                                         ]
                                                     )
                                                 ]
                                             )
                                         ]
                                     )
                                 ]
                             )
                         ]
                     )
                 ]
             )),
            # ('ctemp = 38', [38]),
            # ('feverish @= temp_f > 98.6', True),
            # ('"You " + (feverish ? "have" : "dont have") + " a fever"', "You dont have a fever"),
        ],
    )
    def test_evaluate(self, basic_arithmetic_parser, input_string, expected_value):
        self._test_evaluate(basic_arithmetic_parser, input_string, expected_value)

    @pytest.mark.parametrize(
        "input_string, expected_error_type",
        [
            ("sin()", TypeError),
            ("sin(1, 2)", TypeError),
            ("1/0", ZeroDivisionError),
            ("1000000**1000000", OverflowError),
            ("((0)", ArithmeticParseException),
            ("(((((((((((0)))))))))))", OverflowError),
            ("((((((0)))))))", ArithmeticParseException),
            ("sin({1, 2, 4})", TypeError),
            ("{{{{{{{100}}}}}}}", OverflowError),
        ],
    )
    def test_evaluate_throws_errors(
        self, basic_arithmetic_parser, input_string, expected_error_type
    ):
        with pytest.raises(expected_error_type):
            basic_arithmetic_parser.evaluate(input_string)
            pytest.fail(
                "Exception {} not raised evaluating {!r}".format(
                    expected_error_type.__name__, input_string
                )
            )

    @pytest.mark.parametrize(
        "input_string, expected_value",
        [
            ("1 in (a, b)", False),
            ("1 in [a, b)", True),
            ("1 not in [a, b)", False),
            ("1 ∈ [a, b)", True),
            ("1 ∉ [a, b)", False),
            ("1 in { a, 11, 22, 53}", True),
            ("1 not in {b, 0}", True),
            ("1 in myset", True),
            ("{ 0, 2, 22}", {0, 2, 22}),
            ("{ a, 11, 22, 53} ∩ { 0, 2, 22}", {22,}),
            ("{ a, 11, 22, 53} ∪ { 0, 2, 22}", {0, 1, 2, 11, 22, 53}),
            ("{ a, 11, 22, 53} ∩ {}", set()),
            ("{ a, 11, 22, 53} ∪ {}", {1, 11, 22, 53}),
            ("myset ∩ { 0, 2, 22}", {22, }),
            ("myset ∪ { 0, 2, 22}", {0, 1, 2, 11, 22, 53}),
            ("1 in (myset ∩ { 0, 2, 22})", False),
            ("1 in (myset ∪ { 0, 2, 22})", True),
            ("1 ∈ (myset ∩ { 0, 2, 22})", False),
            ("1 ∉ (myset ∪ { 0, 2, 22})", False),
            ("1 in (myset ∩ {})", False),
            ("1 in (myset ∪ {})", True),
            ("max(aset)", 3),
            ("max({1, 2, 4})", 4),
            ("max({1, 2} ∪ aset)", 3),
            ("min({1, 2, 4})", 1),
        ],
    )
    def test_set_expressions(self, basic_arithmetic_parser, input_string, expected_value):
        basic_arithmetic_parser.parse("a, b = 1, 10")
        basic_arithmetic_parser.parse("myset = { a, 11, 22, 53}")
        basic_arithmetic_parser.parse("aset = {1, 2, 3}")
        self._test_evaluate(basic_arithmetic_parser, input_string, expected_value)

    @pytest.mark.parametrize(
        "input_string, nesting_depth, expected_value, expected_error_type",
        [
            ("((((0))))", 4, 0, None),
            ("(((((0)))))", 4, None, OverflowError),
        ],
    )
    def test_customize_max_expression_depth(self, basic_arithmetic_parser,
                                            input_string, nesting_depth,
                                            expected_value, expected_error_type):
        basic_arithmetic_parser.maximum_expression_depth = nesting_depth

        if expected_error_type is not None:
            with pytest.raises(expected_error_type):
                basic_arithmetic_parser.evaluate(input_string)
                pytest.fail(
                    "Exception {} not raised evaluating {!r}".format(
                        expected_error_type.__name__, input_string
                    )
                )
        else:
            assert basic_arithmetic_parser.evaluate(input_string) == expected_value

    def test_set_parser_vars(self, basic_arithmetic_parser):
        res = []
        expected_x2 = []
        expected_y = []
        for x in range(10):
            basic_arithmetic_parser['x'] = x
            res.append(basic_arithmetic_parser.evaluate("y = x²"))
            expected_x2.append(x * x)
            expected_y.append(basic_arithmetic_parser['y'])

        print(res)
        print(expected_x2)
        print(expected_y)

        assert res == expected_x2 == expected_y

        with pytest.raises(NameError):
            z_value = basic_arithmetic_parser['z']
            pytest.fail("returned unexpected 'z' value {!r}".format(z_value))

    def test_clearing_parser_vars(self, basic_arithmetic_parser):

        with pytest.raises(NameError):
            a_value = basic_arithmetic_parser.evaluate("a")
            pytest.fail("unexpected 'a' value {!r}".format(a_value))

        print("a, b", basic_arithmetic_parser.parse("a, b = 1, 2"))
        print("c", basic_arithmetic_parser.parse("c = a + b"))
        print("clear a", basic_arithmetic_parser.parse("a ="))

        with pytest.raises(NameError):
            a_value = basic_arithmetic_parser["a"]
            pytest.fail("returned unexpected 'a' value {!r}".format(a_value))

        with pytest.raises(NameError):
            a_value = basic_arithmetic_parser.evaluate("c = a + b")
            pytest.fail("unexpected 'a' value {!r}".format(a_value))

    def test_maximum_formula_depth(self, basic_arithmetic_parser):
        basic_arithmetic_parser.maximum_formula_depth = 5
        basic_arithmetic_parser.parse("a @= b + b")
        basic_arithmetic_parser.parse("b @= c + c")
        basic_arithmetic_parser.parse("c @= d + d")
        basic_arithmetic_parser.parse("d @= e + e")
        basic_arithmetic_parser.parse("e @= f + f")

        with pytest.raises(OverflowError):
            basic_arithmetic_parser.parse("f @= g + g")

        basic_arithmetic_parser.parse("f = 1")
        a_value = basic_arithmetic_parser.evaluate("a")
        print(a_value)
        assert a_value == 32

        basic_arithmetic_parser.parse("a, b, c, d, e=")
        basic_arithmetic_parser.parse("a @= b")
        basic_arithmetic_parser.parse("b @= c")
        basic_arithmetic_parser.parse("c @= d")
        basic_arithmetic_parser.parse("d @= e")
        with pytest.raises(OverflowError):
            basic_arithmetic_parser.parse("e @= f")

    def test_max_number_of_vars(self, basic_arithmetic_parser):
        VAR_LIMIT = 20
        basic_arithmetic_parser.max_number_of_vars = VAR_LIMIT

        # compute number of vars that are safe to define by subtracting
        # the number of predefined vars from the allowed limit
        vars_to_define = VAR_LIMIT - len(basic_arithmetic_parser.vars())
        for i in range(vars_to_define):
            basic_arithmetic_parser.evaluate("a{} = 0".format(i))

        # now define one more, which should put us over the limit and raise
        # the exception
        with pytest.raises(Exception):
            basic_arithmetic_parser.evaluate("a{} = 0".format(VAR_LIMIT))
