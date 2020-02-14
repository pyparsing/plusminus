import pytest


@pytest.fixture
def basic_arithmetic_parser():
    from plusminus.plusminus import BasicArithmeticParser
    arith_parser = BasicArithmeticParser()

    # Add temp conversions - you could do this in the test, if you had a specific need,
    # or create a separate/inheriting fixture with these.
    # This also means you could have a descriptive name, then import with much shorter name
    # eg import basic_arithmetic_parser as arith_parser
    arith_parser.initialize_variable("temp_c", "(ftemp - 32) * 5 / 9", as_formula=True)
    arith_parser.initialize_variable("temp_f", "32 + ctemp * 9 / 5", as_formula=True)

    return arith_parser
