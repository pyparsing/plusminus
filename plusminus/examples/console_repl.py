#
# console_repl.py
#
# A simple demonstration REPL around a BasicArithmeticParser.
# NOT FOR PRODUCTION USE.
#
# Copyright 2020, Paul McGuire
#
from pprint import pprint
import textwrap
from plusminus import BasicArithmeticParser

def make_name_list_string(names, indent=''):
    import math
    import itertools
    chunk_size = math.ceil(len(names)**0.5)
    chunks = [list(c) for c in itertools.zip_longest(*[iter(names)] * chunk_size, fillvalue='')]
    col_widths = [max(map(len, chunk)) for chunk in chunks]
    ret = []
    for transpose in zip(*chunks):
        line = indent
        for item, wid in zip(transpose, col_widths):
            line += "{:{}s}".format(item, wid+2)
        ret.append(line.rstrip())
    return '\n'.join(ret)

def usage(parser):
    msg = textwrap.dedent("""\
    Interactive utility to use the plusminus BaseArithmeticParser.

    {parser_usage}    

    Other commands:
    - vars - list all saved variable names
    - clear - clear saved variables
    - quit - quit the REPL
    - help - display this help text
    """)
    print(msg.format(parser_usage=parser.usage()))

def run_repl(parser_class):
    MAX_INPUT_LEN = 100
    done = False
    parser = parser_class()
    while not done:
        cmd = ''
        try:
            cmd = input(">>> ").strip()[:MAX_INPUT_LEN]
        except Exception as input_exc:
            print("invalid input ({})".format(input_exc))
            pass
        if not cmd:
            continue
        elif cmd.lower() == "help":
            usage(parser)
        elif cmd.lower() == "vars":
            pprint(parser.vars(), width=30)
        elif cmd.lower() == "clear":
            parser = parser_class()
        elif cmd.lower() == "quit":
            done = True
        else:
            try:
                result = parser.evaluate(cmd)
            except Exception as e:
                print("{}: {}".format(type(e).__name__, e))
            else:
                print(repr(result))

def main():
    parser_class = BasicArithmeticParser
    run_repl(parser_class)


if __name__ == '__main__':
    import sys
    sys.setrecursionlimit(2000)
    main()
