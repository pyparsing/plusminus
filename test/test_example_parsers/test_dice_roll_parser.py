import pytest

from plusminus.examples.example_parsers import DiceRollParser


@pytest.fixture
def die():
    return DiceRollParser()


class TestSingleRoll:
    def test_single_roll_max_one(self, die):
        assert (
            die.evaluate("d1") == 1
        ), "Single roll of one sided die failed to land on 1."

    @pytest.mark.parametrize(
        "die_max", [2, 4, 5, 10, 100,],
    )
    def test_single_roll_ranges(self, die, die_max):
        test_roll = "d{}".format(die_max)
        roll_value = die.evaluate(test_roll)
        print(test_roll, "->", roll_value)
        assert (
            1 <= roll_value <= die_max
        ), "Die should return value within range of number of sides inclusive."

    def test_single_roll_error_empty_range(self, die):
        with pytest.raises(ValueError):
            die.evaluate("d0")
            pytest.fail("Rolling die with zero sides should raise exception.")


class TestMultipleRolls:
    @pytest.mark.parametrize("die_max", [2, 3, 4, 5, 10, 100,])
    def test_zero_rolls_returns_zero(self, die, die_max):
        test_roll = "0d{}".format(die_max)
        roll_value = die.evaluate(test_roll)
        print(test_roll, "->", roll_value)
        assert roll_value == 0, "Rolling zero times should return 0."

    @pytest.mark.parametrize(
        "num_rolls", [1, 2, 4, 5, 6, 10, 100,],
    )
    def test_multiple_rolls_max_one(self, die, num_rolls):
        test_roll = "{}d1".format(num_rolls)
        roll_value = die.evaluate(test_roll)
        print(test_roll, "->", roll_value)
        assert (
            roll_value == num_rolls
        ), "Incorrect evaluation of multiple die rolls: each roll should equal 1."

    @pytest.mark.parametrize(
        "num_rolls", [1, 2, 4, 5, 10, 100,],
    )
    @pytest.mark.parametrize("die_max", [2, 4, 5, 10, 100,])
    def test_single_roll_ranges(self, die, num_rolls, die_max):
        test_roll = "{}d{}".format(num_rolls, die_max)
        roll_value = die.evaluate(test_roll)
        print(test_roll, "->", roll_value)
        assert (
            num_rolls <= roll_value <= num_rolls * die_max
        ), "Incorrect evaluation of multiple die rolls."


class TestCompoundExamples:
    @pytest.mark.parametrize(
        "evaluation_string, result_min, result_max",
        [
            ("d20+3d4", 1 + 3, 20 + 12),
            ("2d1 + 5d2", 2 + 5, 2 + 10),
            ("5d1 + 12d12 + 6d6", 5 + 12 + 6, 5 + 144 + 36),
            ("2d1 + 0d1 - 6d1*2d1", 2 + 0 - 12, 2 + 0 - 12),
            ("2d1+0d2-7d1+5d1*2d1", 2 + 0 - 7 + 5 * 2, 2 + 0 - 7 + 5 * 2),
            ("5d1*0d12 * 6d6", 5 * 0 * 6, 5 * 0 * 6 * 6),
        ],
    )
    def test_compound_examples(self, die, evaluation_string, result_min, result_max):
        roll_value = die.evaluate(evaluation_string)
        print(evaluation_string, "->", roll_value)
        assert (
            result_min <= roll_value <= result_max
        ), "Incorrect evaluation of {!r}: {}".format(evaluation_string, roll_value)

    def test_rolls_are_distinct(self, die):
        pairs_of_rolls = [die.evaluate("2d6") for _ in range(100)]
        # test for at least 1 odd value, telling us that 2d6 is doing 2 die rolls, not just a single
        # die roll and multiplying by 2
        odd_value_rolled = any(pair % 2 == 1 for pair in pairs_of_rolls)
        assert (
            odd_value_rolled
        ), "Incorrect evaluation of multiple die rolls: no odd results in 100 double rolls."
