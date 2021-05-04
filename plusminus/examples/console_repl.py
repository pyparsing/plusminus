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
from plusminus import ArithmeticParser, ArithmeticParseException


def usage(parser):
    msg = textwrap.dedent(
        """\
    Interactive utility to use the plusminus BaseArithmeticParser.

    {parser_usage}    

    Other commands:
    - vars - list all saved variable names
    - clear - clear saved variables
    - quit - quit the REPL
    - help - display this help text
    """
    )
    print(msg.format(parser_usage=parser.usage()))


def run_repl(parser_class):
    MAX_INPUT_LEN = 100
    done = False
    parser = parser_class()
    parser.user_defined_functions_supported = False
    last_cmd = ""
    while not done:
        cmd = ""
        try:
            cmd = input(">>> ").strip()[:MAX_INPUT_LEN]
        except Exception as input_exc:
            print("invalid input ({})".format(input_exc))
            pass

        if cmd == "redo":
            if last_cmd:
                cmd = last_cmd
            else:
                continue
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
                try:
                    result = parser.evaluate(cmd)
                except NameError:
                    if not cmd.strip().endswith("="):
                        raise
            except ArithmeticParseException as pe:
                print(pe.explain())
            except Exception as e:
                print("{}: {}".format(type(e).__name__, e))
            else:
                if result is not None:
                    print(repr(result))
        last_cmd = cmd


def main():
    parser_class = ArithmeticParser
    run_repl(parser_class)


if __name__ == "__main__":
    import sys

    sys.setrecursionlimit(2000)
    main()
