#
# plusminus_demo.py
#
# short demo of building a repl using different parser classes
#
# Copyright 2020, Paul McGuire
#
from plusminus import ArithmeticParser, BasicArithmeticParser, ArithmeticParseException
from plusminus.examples.example_parsers import *
from pprint import pprint

try:
    import readline
except ImportError:
    readline = None

prompt = '> '
prompt_indent = ' ' * len(prompt)
parser = BasicArithmeticParser()

while True:
    expression = input(prompt).strip()

    if not expression:
        continue

    if expression.lower() == 'quit':
        break

    if expression.lower() == 'help':
        print(parser.usage())
        continue

    if expression.lower() == 'vars':
        pprint(parser.vars())
        continue

    try:
        print(parser.evaluate(expression))
    except ArithmeticParseException as pe:
        print(pe.explain())
    except Exception as exc:
        print("{}: {}".format(type(exc).__name__, exc))
