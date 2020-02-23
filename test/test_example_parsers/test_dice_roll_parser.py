import pytest

from plusminus.examples.example_parsers import DiceRollParser


@pytest.fixture
def die():
    return DiceRollParser()


class TestSingleROll:
    def test_single_roll_max_one(self, die):
        assert die.evaluate('d1') == 1

    @pytest.mark.parametrize('die_max',
                             [2, 4, 5, 10, 100, ],
                             )
    def test_single_roll_ranges(self, die,
                                die_max):
        assert 1 <= die.evaluate(f'd{die_max}') <= die_max

    def test_single_roll_error_empty_range(self, die):
        with pytest.raises(ValueError):
            die.evaluate('d0')


class TestMultipleRolls:
    @pytest.mark.parametrize('die_max',
                             [2, 3, 4, 5, 10, 100, ]
                             )
    def test_zero_rolls_returns_zero(self, die,
                                     die_max):
        assert die.evaluate(f'0d{die_max}') == 0

    @pytest.mark.parametrize('num_rolls',
                             [1, 2, 4, 5, 6, 10, 100, ],
                             )
    def test_multiple_rolls_max_one(self, die,
                                    num_rolls):
        assert num_rolls <= die.evaluate(f'{num_rolls}d1') == num_rolls

    @pytest.mark.parametrize('num_rolls',
                             [1, 2, 4, 5, 10, 100, ],
                             )
    @pytest.mark.parametrize('die_max',
                             [2, 4, 5, 10, 100, ]
                             )
    def test_single_roll_ranges(self, die,
                                num_rolls, die_max):
        assert num_rolls <= die.evaluate(f'{num_rolls}d{die_max}') <= num_rolls*die_max


class TestCompoundExamples:
    @pytest.mark.parametrize('evaluation_string, result_min, result_max',
                             [('d20+3d4', 1+3, 20+12),
                              ('2d1 + 5d2', 2+5, 2+10),
                              ('5d1 + 12d12 + 6d6', 5+12+6, 5+144+36),
                              ('2d1 + 0d1 - 6d1*2d1', 2+0-12, 2+0-12),
                              ('2d1+0d2-7d1+5d1*2d1', 2+0-7+5*2, 2+0-7+5*2),
                              ('5d1*0d12 * 6d6', 5*0*6, 5*0*6*6),
                              ])
    def test_compound_examples(self, die,
                               evaluation_string,
                               result_min, result_max):
        assert result_min <= die.evaluate(evaluation_string) <= result_max

    def test_rolls_are_distinct(self, die):
        pairs_of_rolls = []
        for pair in range(100):
            pairs_of_rolls.append(die.evaluate('2d6'))
        assert any(pair%22 != 0 for pair in pairs_of_rolls)
