# test_datetime_parser.py
import datetime
import pytest

from plusminus.examples.date_time_arithmetic_parser import DateTimeArithmeticParser


@pytest.fixture
def parser():
    ret = DateTimeArithmeticParser()
    # initialize reference timestamp for test calcs
    ret.parse(f"ref = {datetime.datetime(2000, 1, 1).timestamp()}")
    return ret


class TestCalculations:
    @pytest.mark.parametrize(
        "evaluation_string, expected_result",
        [
            ("ref + 3d", (2000, 1, 4)),
            ("ref + 3*(1d + 12h)", (2000, 1, 5, 12, 0, 0)),
            ("str(ref + 3*(1d + 12h))", "2000-01-05 12:00:00"),
            # ("today()", datetime.datetime.today().timetuple()[:3]),
        ],
    )
    def test_calc(self, parser, evaluation_string, expected_result):
        if isinstance(expected_result, tuple):
            expected_timestamp = datetime.datetime(*expected_result).timestamp()
        else:
            expected_timestamp = expected_result
        assert expected_timestamp == parser.evaluate(evaluation_string)


if __name__ == "__main__":
    pytest.main()
