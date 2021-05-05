# test_business_arithmetic_parser.py
import pytest

from plusminus.examples.business_arithmetic_parser import BusinessArithmeticParser


@pytest.fixture
def parser():
    return BusinessArithmeticParser()


class TestCalculations:
    @pytest.mark.parametrize(
        "evaluation_string, expected_result",
        [
            ("25%", 0.25),
            ("20 * 50%", 10),
            ("50% of 20", 10),
            ("20 * (1 - 20%)", 16),
            ("(100 - 20)% of 20", 16),
            ("5 / 20%", 25),
            ("round(FV(20000, 3%, 30), 2)", 48545.25),
            ("round(FV(20000, 3% / 12, 30 * 12), 2)", 49136.84),
        ],
    )
    def test_calc(self, parser, evaluation_string, expected_result):
        assert expected_result == parser.evaluate(evaluation_string)


if __name__ == "__main__":
    pytest.main()
