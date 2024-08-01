import decimal
import math
import pickle

import pytest
import cloudpickle

from plusminus import ArithmeticParser, ArithmeticParseException
import sys

sys.setrecursionlimit(4000)


# Normally this would be kept in a "fixtures.py" file or similar, for access by all test scripts.
@pytest.fixture
def basic_arithmetic_parser():
    arith_parser = ArithmeticParser()

    # Add temp conversions - you could do this in the test, or create a separate/inheriting fixture with these.
    arith_parser.parse("temp_c @= (ftemp - 32) * 5 / 9")
    arith_parser.parse("temp_f @= 32 + ctemp * 9 / 5")
    arith_parser.parse("ctemp = 38")
    arith_parser.parse("feverish @= temp_f > 98.6")

    return arith_parser


# You don't actually need to place test functions in a class,
# but you can setup/scope fixtures for one test/class/module/test_suite run etc
class TestBasicArithmetic:
    def _test_evaluate(self, basic_arithmetic_parser, input_string, expected_value):
        if isinstance(expected_value, (int, float)):
            assert math.isclose(
                basic_arithmetic_parser.evaluate(input_string),
                expected_value,
                abs_tol=1e-12,
            )
        elif isinstance(expected_value, complex):
            observed_value = basic_arithmetic_parser.evaluate(input_string)
            assert math.isclose(
                observed_value.real,
                expected_value.real,
                abs_tol=1e-12,
            ) and math.isclose(
                observed_value.imag,
                expected_value.imag,
                abs_tol=1e-12,
            )

        else:
            assert basic_arithmetic_parser.evaluate(input_string) == expected_value

    @pytest.mark.parametrize(
        "input_string, expected_value",
        [
            ("0**0", 1),
            ("+5", 5),
            ("+(5-3)", 2),
            ("-(5-3)", -2),
            ("-(5-+3)", -2),
            ("-(5--3)", -8),
            ("3**2**3", 3 ** 2 ** 3),
            ('"You" + " win"*3', "You win win win"),
            ("(0)", 0),
            ("((0))", 0),
            ("(((0)))", 0),
            ("((((0))))", 0),
            ("(((((0)))))", 0),
            ("((((((0))))))", 0),
            (
                "{{{{{{100}}}}}}",
                frozenset(
                    [
                        frozenset(
                            [frozenset([frozenset([frozenset([frozenset([100])])])])]
                        )
                    ]
                ),
            ),
            # evaluate formula expressions
            ("ctemp", 38),
            ("temp_f", 100.4),
            (
                '"You " + (feverish ? "have" : "dont have") + " a fever"',
                "You have a fever",
            ),
        ],
    )
    def test_evaluate(self, basic_arithmetic_parser, input_string, expected_value):
        self._test_evaluate(basic_arithmetic_parser, input_string, expected_value)

    @pytest.mark.parametrize(
        "input_string, expected_value",
        [
            ("sin(rad(30))", 0.5),
            ("sin(π/2)", 1.0),
            ("sin(30°)", 0.5),
            ("sin²(30°)", 0.25),
            ("sin³(30°)", 0.125),
            ("cos(30°)", 3**0.5 / 2),
            ("cos²(30°)", 0.75),
            ("cos³(30°)", 3**1.5 / 8),
            ("tan(30°)", 1 / 3**0.5),
            ("tan²(30°)", 1 / 3),
            ("tan³(30°)", 1 / 3**1.5),
            ("deg(sin⁻¹(0.5))", 30),
            ("deg(cos⁻¹(0.5))", 60),
            ("deg(tan⁻¹(1.0))", 45),
            ("deg(tan⁻¹(-1.0))", -45),
            ("log(10)", math.log(10)),
            ("log(10, 2)", math.log(10, 2)),
            ("log(10, 10)", math.log(10, 10)),
            ("log(e)", math.log(math.e)),
            ("hypot(3)", 3),
            ("hypot(3, 4)", 5),
            ("hypot(3, 4, 5, 6) == hypot(3, hypot(4, hypot(5, 6)))", True),
            ("hypot()", 0),
            ("max(10, 11, 12)", 12),
            ("max({10, 11, 12})", 12),
        ],
    )
    def test_evaluate_functions(
        self, basic_arithmetic_parser, input_string, expected_value
    ):
        self._test_evaluate(basic_arithmetic_parser, input_string, expected_value)

    @pytest.mark.parametrize(
        "input_string, expected_value",
        [
            ("²√2", math.sqrt(2)),
            ("2√2", 2 * math.sqrt(2)),
            ("³√2", math.pow(2, 1 / 3)),
            ("3³√2", 3 * math.pow(2, 1 / 3)),
            ("(3³)√2", 3 ** 3 * math.sqrt(2)),
            ("⁹√2", math.pow(2, 1 / 9)),
            ("2⁹√2", 2 * math.pow(2, 1 / 9)),
            ("√-1", (-1) ** (1 / 2)),
        ],
    )
    def test_evaluate_radical_expressions(
        self, basic_arithmetic_parser, input_string, expected_value
    ):
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
            # fmt: off
            ("1 in { a, 11, 22, 53}", True),
            ("1 not in {b, 0}", True),
            ("1 in myset", True),
            ("{ 0, 2, 22}", {0, 2, 22}),
            ("{ a, 11, 22, 53} ∩ { 0, 2, 22}", {22},),
            ("{ a, 11, 22, 53} ∪ { 0, 2, 22}", {0, 1, 2, 11, 22, 53}),
            ("{ a, 11, 22, 53} ∩ {}", set()),
            ("{ a, 11, 22, 53} ∪ {}", {1, 11, 22, 53}),
            ("{ a, 11, 22, 53} & { 0, 2, 22}", {22},),
            ("{ a, 11, 22, 53} | { 0, 2, 22}", {0, 1, 2, 11, 22, 53}),
            ("{ a, 11, 22, 53} & {}", set()),
            ("{ a, 11, 22, 53} | {}", {1, 11, 22, 53}),
            ("myset ∩ { 0, 2, 22}", {22},),
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
            ("{ a, 11, 22, 53} - { 0, 2, 22}", {1, 11, 53}),
            ("{ 0, 31, 1.2} - { 0, 1.2, 31}", set()),
            ("{} - { 0, 2, 22}", set()),
            ("{} ^ {}", set()),
            ("{1, 2, 3} ^ {2, 3, 4}", {1, 4}),
            ("{1, 2, 3} ∆ {2, 3, 4}", {1, 4}),
            # fmt: on
        ],
    )
    def test_set_expressions(
        self, basic_arithmetic_parser, input_string, expected_value
    ):
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
    def test_customize_max_expression_depth(
        self,
        basic_arithmetic_parser,
        input_string,
        nesting_depth,
        expected_value,
        expected_error_type,
    ):
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
        x2_res = []
        y_res = []
        observed_x = []
        expected_x2 = []
        observed_y = []
        for x in range(10):
            basic_arithmetic_parser["x"] = x
            x2_res.append(basic_arithmetic_parser.evaluate("x²"))
            y_res.append(basic_arithmetic_parser.evaluate("y = x²"))
            expected_x2.append(x * x)
            observed_x.append(basic_arithmetic_parser["x"])
            observed_y.append(basic_arithmetic_parser["y"])

        print(x2_res)
        print(y_res)
        print(observed_x)
        print(expected_x2)
        print(observed_y)
        print(basic_arithmetic_parser.vars())

        assert list(range(10)) == observed_x
        assert x2_res == expected_x2
        assert observed_y == expected_x2

        with pytest.raises(NameError):
            z_value = basic_arithmetic_parser["z"]
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
        var_limit = 20
        basic_arithmetic_parser.max_number_of_vars = var_limit

        # compute number of vars that are safe to define by subtracting
        # the number of predefined vars from the allowed limit
        vars_to_define = var_limit - len(basic_arithmetic_parser.vars())
        for i in range(vars_to_define):
            basic_arithmetic_parser.evaluate("a{} = 0".format(i))

        # now define one more, which should put us over the limit and raise
        # the exception
        with pytest.raises(Exception):
            basic_arithmetic_parser.evaluate("a{} = 0".format(var_limit))

    def test_parser_pickling(self, basic_arithmetic_parser):

        pickled_parser = cloudpickle.dumps((basic_arithmetic_parser,))
        parser, = pickle.loads(pickled_parser)

        assert parser["ctemp"] == 38
        parser.parse("ftemp = 212")
        assert parser.evaluate("temp_c") == 100
        assert parser.evaluate("feverish") is True

    def test_parser_pickling2(self):

        parser = ArithmeticParser()
        parser.parse('α = π²')
        parsed_res = parser.parse('β = 90')

        # send parser and results on round trip through cloudpickle
        parsed_pair = cloudpickle.dumps((parser, parsed_res))
        parser, parsed_res = cloudpickle.loads(parsed_pair)

        assert parser["α"] == math.pi ** 2
        assert parser["β"] == 90
        assert parser.evaluate("α * β") == math.pi ** 2 * 90
        assert parser.evaluate("sin(β°)") == 1.0
        assert parsed_res.evaluate() == 90

    def test_decimal_evaluate(self):

        parser = ArithmeticParser(use_decimal=True)

        test_str = "3.000000000000000000001"
        result = parser.evaluate(test_str)
        assert isinstance(result, decimal.Decimal)
        assert str(result) == test_str

        # make sure a pickled parser retains decimal-ness
        parsed_pair = cloudpickle.dumps((parser,))
        parser, = cloudpickle.loads(parsed_pair)

        result = parser.evaluate("100")
        assert isinstance(result, decimal.Decimal)
        assert str(result) == "100"
