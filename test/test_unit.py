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
    @pytest.mark.parametrize(
        "input_string, parse_result",
        [
            ("sin(rad(30))", 0.5),
            ("sin(30°)", 0.5),
            ("sin(π/2)", 1.0),
            ("0**0", 1),
            ("3**2**3", 3 ** 2 ** 3),
            ('"You" + " win"*3', "You win win win"),
            ("100 in [0, 100)", False),
            ("99.9 in [0, 100)", True),
            ("((((((((((0))))))))))", 0),
            # ('ctemp = 38', [38]),
            # ('feverish @= temp_f > 98.6', True),
            # ('"You " + (feverish ? "have" : "dont have") + " a fever"', "You dont have a fever"),
        ],
    )
    def test_evaluate(self, basic_arithmetic_parser, input_string, parse_result):
        assert basic_arithmetic_parser.evaluate(input_string) == parse_result

    @pytest.mark.parametrize(
        "input_string, expected_error_type",
        [
            ("sin()", TypeError),
            ("sin(1, 2)", TypeError),
            ("1/0", ZeroDivisionError),
            ("1000000**1000000", OverflowError),
            ("((0)", ArithmeticParseException),
            ("(((((((((((0)))))))))))", OverflowError),
            ("((((((((((0)))))))))", ArithmeticParseException),
            # ('',),
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
