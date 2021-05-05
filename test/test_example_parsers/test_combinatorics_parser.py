# test_combinatorics_parser.py
import pytest

from plusminus.examples.combinatorics_arithmetic_parser import CombinatoricsArithmeticParser


@pytest.fixture
def parser():
    return CombinatoricsArithmeticParser()


class TestCalculations:
    @pytest.mark.parametrize(
        "evaluation_string, expected_result",
        [
            ("6C1", 6),
            ("6C2", 15),
            ("6C0", 1),
            ("6C6", 1),
            ("6P1", 6),
            ("6P2", 30),
            ("6P0", 1),
            ("6P6", 720),
        ],
    )
    def test_calc(self, parser, evaluation_string, expected_result):
        assert expected_result == parser.evaluate(evaluation_string)


if __name__ == "__main__":
    pytest.main()
