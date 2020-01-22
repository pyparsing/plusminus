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
    Pyrithmetic REPL
    Interactive utility to use the pyrithmetic BaseArithmeticParser.
    
    Enter an arithmetic expression or assignment statement, using the following
    operators:
    {operator_list}
    
    Deferred evaluation assignments can be defined using "@=":
        circle_area @= pi * r**2
    will re-evaluate 'circle_area' using the current value of 'r'.
    
    Expression can include the following functions:
    {function_list}

    Other commands:
    - vars - list all saved variable names
    - clear - clear saved variables
    - quit - quit the REPL
    - help - display this help text
    """)
    func_list = make_name_list_string(names=list({**parser.base_function_map, **parser.added_function_specs}),
                                      indent='  ')
    custom_operators = [str(oper_defn[0]) for oper_defn in parser.added_operator_specs]
    operators = ("** * / mod × ÷ + - < > <= >= == != ≠ ≤ ≥ between-and within-and"
                 " in-range-from-to not and ∧ or ∨ ?:").split()

    oper_list = make_name_list_string(names=custom_operators + operators, indent='  ')
    print(msg.format(function_list=func_list, operator_list=oper_list))

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
