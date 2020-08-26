#
# plusminus.py
#
"""
plusminus

plusminus is a module that builds on the pyparsing infixNotation helper method to build easy-to-code and easy-to-use
parsers for parsing and evaluating infix arithmetic expressions. plusminus's ArithmeticParser class includes
separate parse and evaluate methods, handling operator precedence, override with parentheses, presence or absence of
whitespace, built-in functions, and pre-defined and user-defined variables, functions, and operators.

Copyright 2020, by Paul McGuire

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import namedtuple, deque
from contextlib import contextmanager
from functools import partial, total_ordering
from itertools import groupby
import math
import operator
import random
import pyparsing as pp
import sys

# monkeypatch explain into an instance method
from pyparsing import ParseBaseException, ParseException

if not hasattr(ParseException, "explain_exception"):
    _ParseException = ParseBaseException
    _ParseException.explain = lambda self, fn=ParseException.explain, **kwargs: fn(
        self, depth=0, **kwargs
    )
    ParseException.explain_exception = ParseException.explain
    ParseException.explain = lambda e, *args, **kwargs: _ParseException.explain(
        e, **kwargs
    )

ArithmeticParseException = ParseBaseException

__all__ = """__version__ ArithmeticParser BasicArithmeticParser expressions any_keyword 
             safe_pow safe_str_mult constrained_factorial ArithmeticParseException
             """.split()
__version__ = "0.4.0"

# increase recursion limit if not already modified
if sys.getrecursionlimit() == 1000:
    sys.setrecursionlimit(3000)

ppc = pp.pyparsing_common
pp.ParserElement.enablePackrat()

expressions = {}

# keywords
keywords = {
    k.upper(): pp.Keyword(k)
    for k in """in and or not True False if else mod""".split()
}
vars().update(keywords)
expressions.update(keywords)

any_keyword = pp.MatchFirst(keywords.values())
# noinspection PyUnresolvedReferences
TRUE.addParseAction(lambda: True)
# noinspection PyUnresolvedReferences
FALSE.addParseAction(lambda: False)

FunctionSpec = namedtuple("FunctionSpec", "method arity")

_numeric_type = (int, float, complex)

class PrettySet(frozenset):
    def __repr__(self):
        elems = sorted(self, key=lambda x: id(type(x)))
        sorted_elems = []
        for _, elems_by_type in groupby(elems, key=type):
            sorted_elems.extend(sorted(elems_by_type))
        return "{" + ', '.join(repr(x) for x in sorted_elems) + "}"

PrettySet.__name__ = 'set'

class OperatorString(str):
    def __repr__(self):
        return self


# define special versions of lt, le, etc. to comprehend "is close"
_lt = lambda a, b, eps: (
    a < b and not math.isclose(a, b, abs_tol=eps)
    if isinstance(a, _numeric_type) and isinstance(b, _numeric_type)
    else a < b
)
_le = lambda a, b, eps: (
    a <= b or math.isclose(a, b, abs_tol=eps)
    if isinstance(a, _numeric_type) and isinstance(b, _numeric_type)
    else a <= b
)
_gt = lambda a, b, eps: (
    a > b and not math.isclose(a, b, abs_tol=eps)
    if isinstance(a, _numeric_type) and isinstance(b, _numeric_type)
    else a > b
)
_ge = lambda a, b, eps: (
    a >= b or math.isclose(a, b, abs_tol=eps)
    if isinstance(a, _numeric_type) and isinstance(b, _numeric_type)
    else a >= b
)
_eq = lambda a, b, eps: (
    a == b or math.isclose(a, b, abs_tol=eps)
    if isinstance(a, _numeric_type) and isinstance(b, _numeric_type)
    else a == b
)
_ne = lambda a, b, eps: (
    not math.isclose(a, b, abs_tol=eps)
    if isinstance(a, _numeric_type) and isinstance(b, _numeric_type)
    else a != b
)


def _compute_structure_depth(s, l, t):
    stack = deque([(0, t.asList())])
    max_depth = 0
    while stack:
        depth, node = stack.popleft()
        max_depth = max(max_depth, depth)
        stack.extend((depth+1, item) for item in node if isinstance(item, list))
    return max_depth


_paren_parser = pp.And([pp.nestedExpr()]).addParseAction(_compute_structure_depth)
_brace_parser = pp.And([pp.nestedExpr('{', '}')]).addParseAction(_compute_structure_depth)


def _get_expression_depth(s):
    depths = [t[0] for t, s, e in _paren_parser.scanString(s)]
    if depths:
        return max(depths)
    else:
        return -1


def _get_set_depth(s):
    depths = [t[0] for t, s, e in _brace_parser.scanString(s)]
    if depths:
        return max(depths)
    else:
        return -1


@contextmanager
def _trimming_exception_traceback():
    try:
        yield
    except Exception as e:
        tb = e.__traceback__
        while tb.tb_next:
            tb = tb.tb_next
        e.__traceback__ = tb
        raise e


def collapse_operands(seq, eps=1e-15):
    cur = list(seq)
    last = cur[:]
    while True:
        # print(cur)
        if len(cur) <= 2:
            break
        if _eq(cur[-1], 0, eps):
            cur[-2:] = [1]
            continue
        for i in range(len(cur) - 2, -1, -1):
            if cur[i] == 0:
                # print(i, cur)
                if cur[i + 1] < 0 and (i == len(cur) - 2 or cur[i + 2] % 2 != 0):
                    unused = 0 ** cur[i + 1]
                else:
                    cur[i - 2:] = [1]
                break
        for i in range(len(cur) - 1, 1, -1):
            if cur[i] == 1:
                del cur[i:]
                break
        if cur == last:
            break
        last = cur[:]

    if len(cur) > 1:
        if cur[0] == 0:
            if cur[1] == 0:
                cur[:] = [1]
            else:
                del cur[1:]
        elif cur[1] == 0:
            cur[:] = [1]
        elif cur[0] == 1:
            del cur[1:]

    return cur


def safe_pow(*seq, eps=1e-15):
    operands = collapse_operands(seq, eps)
    ret = 1
    for operand in operands[::-1]:
        op1 = operand  # .evaluate()
        # rough guard against too large values in expression
        if ret == 0:
            ret = 1
        elif op1 == 0:
            if ret > 0:
                ret = 0
            else:
                # raises an exception
                0 ** ret
        elif op1 == 1:
            ret = 1
        elif ret == 1:
            ret = op1
        else:
            if 0 not in (ret, op1) and math.log10(abs(op1)) + math.log10(abs(ret)) > 7:
                raise OverflowError("operands too large for expression")
            ret = op1 ** ret
    return ret


def safe_str_mult(a, b):
    for _ in range(2):
        if isinstance(a, str):
            if b <= 0:
                return ""
            if len(a) * abs(b) > 1e7:
                raise MemoryError("expression creates too large a string")
        a, b = b, a
    return a * b


def constrained_factorial(x):
    if not (0 <= x < 32768):
        raise ValueError("{!r} not in working 0-32,767 range".format(x))
    if math.isclose(x, int(x), abs_tol=1e-12):
        x = int(round(x))
    return math.factorial(x)


@total_ordering
class ArithNode:
    def __init__(self, tokens):
        self.tokens = tokens[0]
        try:
            iter(self.tokens)
        except TypeError:
            self.iterable_tokens = False
        else:
            self.iterable_tokens = not isinstance(self.tokens, str)

    def evaluate(self):
        raise NotImplementedError

    def right_associative_evaluate(self, oper_fn_map):
        pass

    def left_associative_evaluate(self, oper_fn_map):
        pass

    def __repr__(self):
        return (
            type(self).__name__
            + "/"
            + (
                ", ".join(repr(t) for t in self.tokens)
                if self.iterable_tokens
                else repr(self.tokens)
            )
        )

    def __le__(self, other):
        return self.evaluate() <= other.evaluate()

    def __iter__(self):
        return iter(self.tokens)


class LiteralNode(ArithNode):
    def evaluate(self):
        if isinstance(self.tokens, list):
            return tuple(self.tokens)
        else:
            return self.tokens

    def __repr__(self):
        return repr(self.tokens)


class SetNode(ArithNode):
    def evaluate(self):
        return PrettySet(t.evaluate() for t in self.tokens)

    def __repr__(self):
        return "{" + ', '.join(map(repr, self.tokens)) + "}" if self.tokens else "{}"


class UnaryNode(ArithNode):
    def right_associative_evaluate(self, oper_fn_map):
        *opers, operand = self.tokens
        ret = operand.evaluate()
        for op in opers:
            ret = oper_fn_map[op](ret)
        return ret

    def left_associative_evaluate(self, oper_fn_map):
        operand, *opers = self.tokens
        ret = operand.evaluate()
        for op in opers:
            ret = oper_fn_map[op](ret)
        return ret

    def __repr__(self):
        repr_tokens = self.tokens[:]
        for i in range(len(repr_tokens) - 1):
            repr_tokens[i] = OperatorString(repr_tokens[i])
        return "".join(map(repr, repr_tokens))


class BinaryNode(ArithNode):
    def right_associative_evaluate(self, oper_fn_map):
        ret = self.tokens[-1].evaluate()
        for oper, operand in zip(self.tokens[-2::-2], self.tokens[-3::-2]):
            ret = oper_fn_map[oper](operand.evaluate(), ret)
        return ret

    def left_associative_evaluate(self, oper_fn_map):
        ret = self.tokens[0].evaluate()
        for oper, operand in zip(self.tokens[1::2], self.tokens[2::2]):
            ret = oper_fn_map[oper](ret, operand.evaluate())
        return ret

    def __repr__(self):
        repr_tokens = self.tokens[:]
        for i in range(1, len(repr_tokens), 2):
            repr_tokens[i] = OperatorString(repr_tokens[i])
        return "({})".format(" ".join(map(repr, repr_tokens)))


class TernaryNode(ArithNode):
    opns_map = {}

    def left_associative_evaluate(self, oper_fn_map):
        operands = self.tokens
        ret = operands[0].evaluate()
        i = 1
        while i < len(operands):
            op1, operand1, op2, operand2 = operands[i: i + 4]
            ret = oper_fn_map[op1, op2](ret, operand1.evaluate(), operand2.evaluate())
            i += 4
        return ret

    # eval logic is the same for left and right assoc ternary expressions
    right_associative_evaluate = left_associative_evaluate

    def evaluate(self):
        with _trimming_exception_traceback():
            return self.left_associative_evaluate(self.opns_map)

    def __repr__(self):
        repr_tokens = self.tokens[:]
        for i in range(1, len(repr_tokens), 2):
            repr_tokens[i] = OperatorString(repr_tokens[i])
        return "({})".format(" ".join(map(repr, repr_tokens)))


class ArithmeticParser:
    """
    Base class for defining arithmetic parsers.
    """

    LEFT = pp.opAssoc.LEFT
    RIGHT = pp.opAssoc.RIGHT

    def usage(self):
        import textwrap

        def make_name_list_string(names, indent=""):
            import itertools

            def unique(seq):
                seen = set()
                for obj in seq:
                    if obj not in seen:
                        seen.add(obj)
                        yield obj

            names = list(unique(names))
            chunk_size = -int(-len(names) ** 0.5 // 1)
            chunks = [
                list(c)
                for c in itertools.zip_longest(
                    *[iter(names)] * chunk_size, fillvalue=""
                )
            ]
            col_widths = [max(map(len, chunk)) for chunk in chunks]
            ret = []
            for transpose in zip(*chunks):
                line = indent
                for item, wid in zip(transpose, col_widths):
                    line += "{:{}s}".format(item, wid + 2)
                ret.append(line.rstrip())
            return "\n".join(ret)

        msg = textwrap.dedent(
            """\
        Enter an arithmetic expression or assignment statement, using
        the following operators:
        {operator_list}

        Multiple assignments can be made using lists of variable names and
        corresponding lists of expressions (lists must be of matching lengths).
            x₁, y₁ = 1, 2
            a, b, c = 1, 2, a+b

        Deferred evaluation assignments can be defined using "@=":
            circle_area @= pi * r**2
        will re-evaluate 'circle_area' using the current value of 'r'.

        Expression can include the following functions:
        {function_list}
        """
        )
        func_list = make_name_list_string(
            names=list({**self.base_function_map, **self.added_function_specs}),
            indent="  ",
        )
        custom_operators = [
            str(oper_defn[0]) for oper_defn in self.added_operator_specs
        ]
        operators = self.base_operators

        oper_list = make_name_list_string(
            names=custom_operators + operators + ["|absolute-value|"], indent="  "
        )
        return msg.format(function_list=func_list, operator_list=oper_list)

    class ArithmeticUnaryOp(UnaryNode):
        opns_map = {
            "+": lambda x: x,
            "-": operator.neg,
            "−": operator.neg,
        }

        def evaluate(self):
            with _trimming_exception_traceback():
                return self.right_associative_evaluate(self.opns_map)

    class ArithmeticUnaryPostOp(UnaryNode):
        opns_map = {}

        def evaluate(self):
            with _trimming_exception_traceback():
                return self.left_associative_evaluate(self.opns_map)

    class ArithmeticBinaryOp(BinaryNode):
        opns_map = {
            "+": operator.add,
            "-": operator.sub,
            "−": operator.sub,
            "*": safe_str_mult,
            "/": operator.truediv,
            "//": operator.floordiv,
            "mod": operator.mod,
            "×": safe_str_mult,
            "÷": operator.truediv,
        }

        def evaluate(self):
            with _trimming_exception_traceback():
                return self.left_associative_evaluate(self.opns_map)

    class ExponentBinaryOp(ArithmeticBinaryOp):
        def evaluate(self):
            with _trimming_exception_traceback():
                # parsed left-to-right, but evaluate right-to-left
                operands = [t.evaluate() for t in self.tokens[::2]]
                if not all(isinstance(op, (int, float, complex)) for op in operands):
                    raise TypeError("invalid operators for exponentiation")

                return safe_pow(*operands)

    class IdentifierNode(ArithNode):
        _assigned_vars = {}

        @property
        def name(self):
            return self.tokens

        def evaluate(self):
            with _trimming_exception_traceback():
                if self.name in self._assigned_vars:
                    return self._assigned_vars[self.name].evaluate()
                else:
                    raise NameError("variable {!r} not known".format(self.name))

        def __repr__(self):
            return self.name

    class ArithmeticFunction(ArithNode):
        fn_map = None
        def evaluate(self):
            with _trimming_exception_traceback():
                fn_name, *fn_args = self.tokens
                if fn_name not in self.fn_map:
                    raise ValueError(
                        "{!r} is not a recognized function".format(fn_name)
                    )
                fn_spec = self.fn_map[fn_name]

                if isinstance(fn_spec.arity, tuple):
                    if not any(arit == len(fn_args) for arit in fn_spec.arity):
                        raise TypeError(
                    "{} takes {} {}, {} given".format(
                            fn_name,
                            " or ".join(str(ari) for ari in fn_spec.arity),
                            ("arg", "args")[fn_spec.arity[-1] != 1],
                            len(fn_args),
                        )
                    )

                elif fn_spec.arity not in (len(fn_args), ...):
                    raise TypeError(
                        "{} takes {} {}, {} given".format(
                            fn_name,
                            fn_spec.arity,
                            ("arg", "args")[fn_spec.arity != 1],
                            len(fn_args),
                        )
                    )
                return fn_spec.method(*[arg.evaluate() for arg in fn_args])

        def __repr__(self):
            return "{}({})".format(self.tokens[0], ", ".join(map(repr, self.tokens[1:])))

    def __init__(self):
        self.max_number_of_vars = 1000
        self.max_var_memory = 10 ** 6

        self._added_operator_specs = []
        self._added_function_specs = {}
        self._base_operators = (
            "** * // / mod × ÷ + - < > <= >= == != ≠ ≤ ≥ ∈ ∉ ∩ ∪ in not and ∧ or ∨ ?:"
        ).split()
        self._base_function_map = {
            "abs": FunctionSpec(abs, 1),
            "round": FunctionSpec(round, (1, 2)),
            "trunc": FunctionSpec(math.trunc, 1),
            "ceil": FunctionSpec(math.ceil, 1),
            "floor": FunctionSpec(math.floor, 1),
            "min": FunctionSpec(min, ...),
            "max": FunctionSpec(max, ...),
            "str": FunctionSpec(lambda x: str(x), 1),
            "bool": FunctionSpec(lambda x: not not x, 1),
        }

        # epsilon for computing "close" floating point values - can be updated in customize
        self.epsilon = 1e-15

        # customize can update or replace with different characters
        self.ident_letters = (
            pp.srange("[A-Za-z]")
            + pp.srange("[ªºÀ-ÖØ-öø-ÿ]")
            + pp.srange("[Α-Ωα-ω]")
        )

        # customize can raise or lower the maximum expression depth
        # to be supported - default = 8
        self.maximum_expression_depth = 6
        self.maximum_formula_depth = 12
        self.maximum_set_depth = 6

        # storage for assigned variables
        self._variable_map = {}

        # customize can add pre-defined constants
        self._initial_variables = {}

        self.customize()
        self._parser = self.make_parser()

    @property
    def base_function_map(self):
        return {**self._base_function_map}

    @property
    def added_function_specs(self):
        return {**self._added_function_specs}

    @property
    def base_operators(self):
        return self._base_operators[:]

    @property
    def added_operator_specs(self):
        return self._added_operator_specs[:]

    def scanString(self, *args):
        yield from self.get_parser().scanString(*args)

    def parse(self, *args, **kwargs):
        """Parses an expression."""
        if _get_expression_depth(args[0]) > self.maximum_expression_depth:
            raise OverflowError("expression too deeply nested")

        if _get_set_depth(args[0]) > self.maximum_set_depth:
            raise OverflowError("set too deeply nested")

        parsed = self.get_parser().parseString(*args, **kwargs)

        if parsed:
            return parsed[0]

    def evaluate(self, arith_expression):
        """Evaluates an expression and returns its result."""
        with _trimming_exception_traceback():
            parsed = self.parse(arith_expression, parseAll=True)
            return parsed.evaluate()

    def __getattr__(self, attr):
        parser = self._parser
        if hasattr(parser, attr):
            return getattr(parser, attr)
        raise AttributeError("no such attribute {!r}".format(attr))

    def __getitem__(self, key):
        self_vars = self.vars()
        if key in self_vars:
            return self_vars[key]
        else:
            raise NameError("no such variable {!r}".format(key))

    def __delitem__(self, key):
        self_vars = self._variable_map
        if key in self_vars:
            del self_vars[key]

    def __iter__(self):
        raise NotImplementedError

    def __setitem__(self, name, value):
        self._variable_map[name] = LiteralNode([value])

    def customize(self):
        pass

    def add_operator(self, operator_expr, arity, assoc, parse_action):
        operator_node_superclass = {
            (1, pp.opAssoc.LEFT): self.ArithmeticUnaryPostOp,
            (1, pp.opAssoc.RIGHT): self.ArithmeticUnaryOp,
            (2, pp.opAssoc.LEFT): self.ArithmeticBinaryOp,
            (2, pp.opAssoc.RIGHT): self.ArithmeticBinaryOp,
            (3, pp.opAssoc.LEFT): TernaryNode,
            (3, pp.opAssoc.RIGHT): TernaryNode,
        }[arity, assoc]
        operator_node_class = type(
            "",
            (operator_node_superclass,),
            {"opns_map": {str(operator_expr): parse_action}},
        )
        self._added_operator_specs.insert(
            0, (operator_expr, arity, assoc, operator_node_class)
        )

    def initialize_variable(self, vname, vvalue, as_formula=False):
        self._initial_variables[vname] = (vvalue, as_formula)

    def add_function(self, fn_name, fn_arity, fn_method):
        self._added_function_specs[fn_name] = FunctionSpec(fn_method, fn_arity)

    def get_parser(self):
        if self._parser is None:
            self._parser = self.make_parser()
        return self._parser

    def make_parser(self):
        arith_operand = pp.Forward()
        LPAR, RPAR, LBRACK, RBRACK, LBRACE, RBRACE, COMMA = map(pp.Suppress, "()[]{},")
        fn_name_expr = pp.Word(
            "_" + self.ident_letters, "_" + self.ident_letters + pp.nums
        )
        function_expression = pp.Group(
            fn_name_expr("fn_name")
            + LPAR
            + pp.Optional(pp.delimitedList(arith_operand))("args")
            + RPAR
        )
        function_node_class = type(
            "Function",
            (self.ArithmeticFunction,),
            {"fn_map": {**self._base_function_map, **self._added_function_specs}},
        )
        function_expression.addParseAction(function_node_class)

        range_punc = ((LPAR() | RPAR()).setParseAction(lambda: False)
                      | (LBRACK() | RBRACK()).setParseAction(lambda: True))
        range_expression = pp.Group(
            range_punc("lower_inclusive")
            + arith_operand("lower")
            + COMMA
            + arith_operand("upper")
            + range_punc("upper_inclusive")
        )
        set_operand = pp.Group(LBRACE
                               + pp.Optional(pp.delimitedList(arith_operand))
                               + RBRACE).addParseAction(SetNode)

        numeric_operand = ppc.number().addParseAction(LiteralNode)
        qs = pp.QuotedString('"', escChar="\\") | pp.QuotedString("'", escChar="\\")
        string_operand = qs.addParseAction(LiteralNode)
        bool_operand = (TRUE | FALSE).addParseAction(LiteralNode)

        ident_sub_chars = pp.srange("[ₐ-ₜ]") + pp.srange("[₀-₉]")
        var_name = pp.Combine(
            ~any_keyword
            + pp.Word("_" + self.ident_letters, "_" + self.ident_letters + pp.nums)
            + pp.Optional(pp.Word(ident_sub_chars))
        ).setName("identifier")

        class BinaryComparison(BinaryNode):
            def __init__(self, tokens):
                super().__init__(tokens)
                self.epsilon = 1e-15

            opns_map = {
                "<": partial(_lt, eps=self.epsilon),
                "<=": partial(_le, eps=self.epsilon),
                ">": partial(_gt, eps=self.epsilon),
                ">=": partial(_ge, eps=self.epsilon),
                "==": partial(_eq, eps=self.epsilon),
                "!=": partial(_ne, eps=self.epsilon),
                "≠": partial(_ne, eps=self.epsilon),
                "≤": partial(_le, eps=self.epsilon),
                "≥": partial(_ge, eps=self.epsilon),
            }

            def evaluate(self):
                with _trimming_exception_traceback():
                    return self.left_associative_evaluate(self.opns_map)

            def left_associative_evaluate(self, oper_fn_map):
                with _trimming_exception_traceback():
                    last = self.tokens[0].evaluate()
                    ret = True
                    for oper, operand in zip(self.tokens[1::2], self.tokens[2::2]):
                        next_ = operand.evaluate()
                        ret = ret and oper_fn_map[oper](last, next_)
                        last = next_
                    return ret

        class UnaryNot(UnaryNode):
            def evaluate(self):
                with _trimming_exception_traceback():
                    return self.right_associative_evaluate({"not": operator.not_})

        class InRangeNode(UnaryNode):
            def evaluate(self):
                nonlocal identifier_node_class
                operand, op, range_expr = self.tokens
                assert_negate_fn = (lambda x: not not x, lambda x: not x)[op in ('∉', 'not_in')]
                if isinstance(range_expr, (identifier_node_class, SetBinaryOp)):
                    range_expr = range_expr.evaluate()
                if isinstance(range_expr, (PrettySet, SetNode)):
                    with _trimming_exception_traceback():
                        op_val = operand.evaluate()
                        return assert_negate_fn(
                            sum(op_val == elem for elem in range_expr.evaluate())
                                if isinstance(range_expr, SetNode) else op_val in range_expr
                        )
                elif 'lower_inclusive' in range_expr:
                    range_fn = {
                        (False, False): lambda a, b, c: a < b < c,
                        (False, True): lambda a, b, c: a < b <= c,
                        (True, False): lambda a, b, c: a <= b < c,
                        (True, True): lambda a, b, c: a <= b <= c,
                    }[range_expr.lower_inclusive, range_expr.upper_inclusive]

                    with _trimming_exception_traceback():
                        return assert_negate_fn(range_fn(range_expr.lower.evaluate(),
                                                operand.evaluate(),
                                                range_expr.upper.evaluate()))
                else:
                    with _trimming_exception_traceback():
                        return assert_negate_fn(operand.evaluate() in range_expr.evaluate())

            def __repr__(self):
                operand, op, range_expr = self.tokens

                lower_symbol = "(["[range_expr.lower_inclusive]
                upper_symbol = ")]"[range_expr.upper_inclusive]
                repr_format = "{} {} {}{}, {}{}"
                return repr_format.format(repr(operand),
                                          op,
                                          lower_symbol,
                                          repr(range_expr.lower),
                                          repr(range_expr.upper),
                                          upper_symbol
                                          )

        class BinaryComp(BinaryNode):
            opns_map = {
                "and": operator.and_,
                "or": operator.or_,
                "∧": operator.and_,
                "∨": operator.or_,
            }

            def evaluate(self):
                with _trimming_exception_traceback():
                    return self.left_associative_evaluate(self.opns_map)

            def left_associative_evaluate(self, oper_fn_map):
                with _trimming_exception_traceback():
                    last = bool(self.tokens[0].evaluate())
                    ret = True
                    for oper, operand in zip(self.tokens[1::2], self.tokens[2::2]):
                        next_ = bool(operand.evaluate())
                        ret = ret and oper_fn_map[oper](last, next_)
                        if not ret:
                            break
                        last = next_
                    return ret

        class TernaryComp(TernaryNode):
            opns_map = {
                ("?", ":"): (lambda a, b, c: b if a else c),
            }

        class RoundToEpsilon:
            def __init__(self, result):
                self._result = result
                self.epsilon = 1e-15

            def __repr__(self):
                return "~" + str(self._result)

            def evaluate(self):
                with _trimming_exception_traceback():
                    # print(self._result.dump())
                    ret = self._result[0].evaluate()
                    if isinstance(ret, (float, complex)):
                        if math.isclose(ret.imag, 0, abs_tol=self.epsilon):
                            ret = round(ret.real, 15)
                        if math.isclose(ret.real, 0, abs_tol=self.epsilon):
                            if ret.imag:
                                ret = complex(0, ret.imag)
                            else:
                                ret = 0
                        if (
                            not isinstance(ret, complex)
                            and abs(ret) < 1e15
                            and math.isclose(ret, int(ret), abs_tol=self.epsilon)
                        ):
                            return int(ret)
                    return ret

        identifier_node_class = type(
            "Identifier", (self.IdentifierNode,), {"_assigned_vars": self._variable_map}
        )
        var_name.addParseAction(identifier_node_class)

        def set_intersection(a, b):
            a_set = a if isinstance(a, PrettySet) else PrettySet(elem.evaluate() for elem in a)
            b_set = b if isinstance(b, PrettySet) else PrettySet(elem.evaluate() for elem in b)
            return PrettySet(a_set.intersection(b_set))

        def set_union(a, b):
            a_set = a if isinstance(a, PrettySet) else PrettySet(elem.evaluate() for elem in a)
            b_set = b if isinstance(b, PrettySet) else PrettySet(elem.evaluate() for elem in b)
            return PrettySet(a_set.union(b_set))

        class SetBinaryOp(BinaryNode):
            opns_map = {
                "∩": set_intersection,
                "∪": set_union,
            }

            def evaluate(self):
                with _trimming_exception_traceback():
                    return self.left_associative_evaluate(self.opns_map)

        set_expression = pp.infixNotation(set_operand | var_name, [
            ("∩", 2, pp.opAssoc.LEFT, SetBinaryOp),
            ("∪", 2, pp.opAssoc.LEFT, SetBinaryOp),
            ])

        # noinspection PyUnresolvedReferences
        NOT_IN = (NOT() + IN()).addParseAction('_'.join)
        base_operator_specs = [
            ("**", 2, pp.opAssoc.LEFT, self.ExponentBinaryOp),
            (pp.oneOf("- −"), 1, pp.opAssoc.RIGHT, self.ArithmeticUnaryOp),
            (pp.oneOf("* // / mod × ÷"), 2, pp.opAssoc.LEFT, self.ArithmeticBinaryOp),
            (pp.oneOf("+ - −"), 2, pp.opAssoc.LEFT, self.ArithmeticBinaryOp),
            (pp.oneOf("< > <= >= == != ≠ ≤ ≥"), 2, pp.opAssoc.LEFT, BinaryComparison),
            ((IN | NOT_IN | pp.oneOf("∈ ∉")) - (range_expression | set_expression | var_name), 1, pp.opAssoc.LEFT,
             InRangeNode),
            (NOT, 1, pp.opAssoc.RIGHT, UnaryNot),
            (AND | "∧", 2, pp.opAssoc.LEFT, BinaryComp),
            (OR | "∨", 2, pp.opAssoc.LEFT, BinaryComp),
            (("?", ":"), 3, pp.opAssoc.RIGHT, TernaryComp),
        ]
        ABS_VALUE_VERT = pp.Suppress("|")
        abs_value_expression = ABS_VALUE_VERT + arith_operand + ABS_VALUE_VERT

        def cvt_to_function_call(tokens):
            ret = pp.ParseResults(["abs"]) + tokens
            ret["fn_name"] = "abs"
            ret["args"] = tokens
            return [ret]

        abs_value_expression.addParseAction(cvt_to_function_call, function_node_class)

        arith_operand <<= pp.infixNotation(
            (
                function_expression
                | abs_value_expression
                | string_operand
                | numeric_operand
                | bool_operand
                | var_name
                | set_expression
            ),
            self._added_operator_specs + base_operator_specs,
        )
        rvalue = (arith_operand ^ set_expression).setName("rvalue")
        rvalue.setName("arithmetic expression")
        lvalue = var_name()

        value_assignment_statement = (
            pp.delimitedList(lvalue)("lhs")
            + pp.oneOf("<- =")
            + pp.delimitedList(rvalue)("rhs")
        )

        value_clear_statement = (
            pp.delimitedList(lvalue)("lhs")
            + pp.oneOf("<- =")
            + pp.StringEnd()
        )

        def eval_and_store_value(tokens):
            if len(tokens.lhs) > len(tokens.rhs):
                raise TypeError("not enough values given")
            if len(tokens.lhs) < len(tokens.rhs):
                raise TypeError("not enough variable names given")

            assignments = []
            assigned_vars = identifier_node_class._assigned_vars
            for lhs_name, rhs_expr in zip(tokens.lhs, tokens.rhs):
                rval = LiteralNode([rhs_expr.evaluate()])
                var_name = lhs_name.name
                if (
                    var_name not in assigned_vars
                    and len(assigned_vars) >= self.max_number_of_vars
                ):
                    raise Exception("too many variables defined")
                assigned_vars[var_name] = rval
                assignments.append(rval)
            return LiteralNode([assignments]) if len(assignments) > 1 else rval

        value_assignment_statement.addParseAction(eval_and_store_value)
        value_assignment_statement.setName("assignment statement")

        formula_assignment_operator = pp.Literal("@=")
        formula_assignment_statement = (
            lvalue("lhs") + formula_assignment_operator + rvalue("rhs")
        )
        formula_assignment_statement.setName("formula statement")

        def verify_formula_not_recursive(tokens):
            # see if this var_name is recursively referenced
            formula_defn = tokens.rhs
            dest_var_name = tokens.lhs.name

            to_visit = deque([formula_defn])
            while to_visit:
                cur_expr = to_visit.popleft()
                if isinstance(cur_expr, self.IdentifierNode):
                    if cur_expr.name == dest_var_name:
                        raise Exception("illegal recursion, {!r} is used in expression".format(dest_var_name))

                    cur_expr = self._variable_map.get(cur_expr.name)
                    if cur_expr is not None:
                        to_visit.append(cur_expr)
                    continue

                if isinstance(cur_expr, (str, LiteralNode)):
                    continue

                try:
                    for e in cur_expr:
                        to_visit.append(e)
                except TypeError:
                    continue

        def store_parsed_value(tokens):
            def get_depth(formula_node):
                max_depth = 0
                to_visit = deque([(1, formula_node)])
                while to_visit:
                    cur_depth, cur_expr = to_visit.popleft()
                    max_depth = max(max_depth, cur_depth)

                    if isinstance(cur_expr, self.IdentifierNode):
                        cur_expr = self._variable_map.get(cur_expr.name)
                        if cur_expr is not None:
                            to_visit.append((cur_depth + 1, cur_expr))
                        continue

                    if isinstance(cur_expr, (str, LiteralNode)):
                        continue

                    try:
                        for e in cur_expr:
                            to_visit.append((cur_depth, e))
                    except TypeError:
                        continue

                return max_depth

            # check if any formulas exceed maximum allowed depth
            rval = tokens.rhs
            dest_var_name = tokens.lhs.name
            assigned_vars = identifier_node_class._assigned_vars
            if (
                dest_var_name not in assigned_vars
                    and len(assigned_vars) >= self.max_number_of_vars
                or sum(sys.getsizeof(vv) for vv in assigned_vars.values()) > self.max_var_memory
            ):
                raise Exception("too many variables defined")
            assigned_vars[dest_var_name] = rval

            # verify that no expressions exceed max depth
            for formula_name, formula_defn in assigned_vars.items():
                if isinstance(formula_defn, (str, LiteralNode)):
                    continue

                formula_depth = get_depth(formula_defn)
                if formula_depth > self.maximum_formula_depth:
                    assigned_vars.pop(dest_var_name, None)
                    raise OverflowError("function variables nested too deeply")

            return rval

        def clear_parsed_value(tokens):
            assigned_vars = identifier_node_class._assigned_vars
            for var in tokens.lhs:
                assigned_vars.pop(var.name, None)
            return tokens.lhs.asList()

        value_assignment_statement.addParseAction(RoundToEpsilon)
        lone_rvalue = rvalue().addParseAction(RoundToEpsilon)
        formula_assignment_statement.addParseAction(verify_formula_not_recursive)
        formula_assignment_statement.addParseAction(store_parsed_value)
        value_clear_statement.addParseAction(clear_parsed_value)

        parser = value_assignment_statement | value_clear_statement | formula_assignment_statement | lone_rvalue

        # init _variable_map with any pre-defined values
        for varname, (varvalue, as_formula) in self._initial_variables.items():
            if as_formula:
                try:
                    parser.parseString(
                        "{} {} {}".format(
                            varname, formula_assignment_operator, varvalue
                        )
                    )
                except NameError:
                    pass
            else:
                self._variable_map[varname] = LiteralNode([varvalue])
        return parser

    def vars(self):
        ret = {}
        for k, v in self._variable_map.items():
            if isinstance(v, LiteralNode):
                ret[k] = v.evaluate()
            else:
                ret[k] = repr(v)
        return ret


def log(x, y=10):
    if math.isclose(y, 2, abs_tol=1e-15):
        return math.log2(x)
    if math.isclose(y, 10, abs_tol=1e-15):
        return math.log10(x)
    return math.log(x, y)


class BasicArithmeticParser(ArithmeticParser):
    """
    Ready to use, basic parser. Example:
    ```python
    >>> parser = BasicArithmeticParser()
    >>> parser.evaluate("gamma(2)") # Predefined functions
    1
    >>> parser.evaluate("pi*9") # Predefined variables
    28.274333882308138
    >>> parser.evaluate("9**3")
    729
    ```
    """
    def customize(self):
        import math

        super().customize()
        phi = (1.0 + 5**0.5) / 2.0   # The golden number

        self.initialize_variable("pi", math.pi)
        self.initialize_variable("π", math.pi)
        self.initialize_variable("τ", math.tau)
        self.initialize_variable("tau", math.tau)
        self.initialize_variable("e", math.e)
        self.initialize_variable("φ", phi)
        self.initialize_variable("ϕ", phi)
        self.initialize_variable("phi", phi)
        self.add_function("sin", 1, math.sin)
        self.add_function("cos", 1, math.cos)
        self.add_function("tan", 1, math.tan)
        self.add_function("asin", 1, math.asin)
        self.add_function("acos", 1, math.acos)
        self.add_function("atan", 1, math.atan)
        self.add_function("sinh", 1, math.sinh)
        self.add_function("cosh", 1, math.cosh)
        self.add_function("tanh", 1, math.tanh)
        self.add_function("rad", 1, math.radians)
        self.add_function("deg", 1, math.degrees)
        self.add_function("ln", 1, lambda x: math.log(x))
        self.add_function("log", (1, 2), math.log) # log function can accept one or two values
        self.add_function("log2", 1, math.log2)
        self.add_function("log10", 1, math.log10)
        self.add_function("gcd", 2, math.gcd)
        self.add_function(
            "lcm", 2,
            lambda a, b: abs(a*b) // math.gcd(a, b) if a or b else 0
        )
        self.add_function("gamma", 1, math.gamma)
        self.add_function("hypot", ..., lambda *seq: sum(safe_pow(i, 2) for i in seq)**0.5)
        self.add_function("rnd", 0, random.random)
        self.add_function("randint", 2, random.randint)
        self.add_function("sgn", 1, lambda x: 0 if _eq(x, 0, self.epsilon) else 1 if x > 0 else -1),
        self.add_operator("°", 1, ArithmeticParser.LEFT, math.radians)
        # avoid clash with '!=' operator
        factorial_operator = (~pp.Literal("!=") + "!").setName("!")
        self.add_operator(
            factorial_operator, 1, ArithmeticParser.LEFT, constrained_factorial
        )
        self.add_operator("⁻¹", 1, ArithmeticParser.LEFT, lambda x: 1 / x)
        self.add_operator("²", 1, ArithmeticParser.LEFT, lambda x: safe_pow(x, 2))
        self.add_operator("³", 1, ArithmeticParser.LEFT, lambda x: safe_pow(x, 3))
        self.add_operator("√", 1, ArithmeticParser.RIGHT, lambda x: x ** 0.5)
        self.add_operator("√", 2, ArithmeticParser.LEFT, lambda x, y: x * y ** 0.5)
