#
# plusminus.py
#
"""
plusminus

plusminus is a module that builds on the pyparsing infix_notation helper method to build easy-to-code and easy-to-use
parsers for parsing and evaluating infix arithmetic expressions. plusminus's BaseArithmeticParser class includes
separate parse and evaluate methods, handling operator precedence, override with parentheses, presence or absence of
whitespace, built-in functions, and pre-defined and user-defined variables, functions, and operators.

Copyright 2020-2022, by Paul McGuire

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
import decimal
from abc import ABC
from collections import namedtuple, deque
from contextlib import contextmanager
from functools import partial, total_ordering
from itertools import groupby
import math
import operator
import random
import string
import sys
from typing import Dict, Callable, List, Any, Union, Tuple

import pyparsing as pp

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

__all__ = """__version__ __version_info__ ArithmeticParser BaseArithmeticParser expressions any_keyword 
             safe_pow safe_str_mult constrained_factorial ArithmeticParseException log
             DEFAULT_BASE_FUNCTION_MAP""".split()

VersionInfo = namedtuple("VersionInfo", "major minor micro releaselevel serial")
__version_info__ = VersionInfo(0, 8, 0, "final", 0)
__version__ = ".".join(map(str, __version_info__[:3]))

# increase recursion limit if not already modified
if sys.getrecursionlimit() == 1000:
    sys.setrecursionlimit(3000)

ppc = pp.pyparsing_common
pp.ParserElement.enablePackrat()

expressions = {}

# keywords
keywords = {
    k.upper(): pp.Keyword(k) for k in """in and or not True False if else mod""".split()
}
vars().update(keywords)
expressions.update(keywords)

any_keyword = pp.MatchFirst(keywords.values())
# noinspection PyUnresolvedReferences
TRUE.addParseAction(lambda: True)
# noinspection PyUnresolvedReferences
FALSE.addParseAction(lambda: False)

FunctionSpec = namedtuple("FunctionSpec", "name method arity")
FunctionSpec.__str__ = lambda self: self.name
OperatorSpec = namedtuple("OperatorSpec", "op_expr arity assoc action")
OperatorSpec.__str__ = lambda self: str(self.op_expr)

_numeric_type = (int, float, complex)


class PrettySet(frozenset):
    def __repr__(self):
        elems = sorted(self, key=lambda x: id(type(x)))
        sorted_elems = []
        for _, elems_by_type in groupby(elems, key=type):
            sorted_elems.extend(sorted(elems_by_type))
        return "{" + ", ".join(repr(x) for x in sorted_elems) + "}"


PrettySet.__name__ = "set"


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


def _make_name_list_string(names, indent=""):
    import itertools

    def unique(seq):
        seen = set()
        for obj in seq:
            if obj not in seen:
                seen.add(obj)
                yield obj

    def split_alternatives(seq):
        """
        If any expressions are MatchFirst's or Or's, list them separately
        """
        for s in seq:
            if " | " in s:
                yield from s.split(" | ")
            elif " ^ " in s:
                yield from s.split(" ^ ")
            else:
                yield s

    names = list(unique(split_alternatives(names)))
    chunk_size = -int(-len(names) ** 0.5 // 1)
    chunks = [
        list(c)
        for c in itertools.zip_longest(*[iter(names)] * chunk_size, fillvalue="")
    ]
    col_widths = [max(map(len, chunk)) for chunk in chunks]
    ret = []
    for transpose in zip(*chunks):
        line = indent
        for item, wid in zip(transpose, col_widths):
            line += f"{item:{wid + 2}s}"
        ret.append(line.rstrip())
    return "\n".join(ret)


def _compute_structure_depth(s, l, t):
    stack = deque([(0, t.asList())])
    max_depth = 0
    while stack:
        depth, node = stack.popleft()
        max_depth = max(max_depth, depth)
        stack.extend((depth + 1, item) for item in node if isinstance(item, list))
    return max_depth


_paren_parser = pp.And([pp.nestedExpr()]).addParseAction(_compute_structure_depth)
_brace_parser = pp.And([pp.nestedExpr("{", "}")]).addParseAction(
    _compute_structure_depth
)


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


def _collapse_operands(seq, eps=1e-15):
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
                    cur[i - 2 :] = [1]
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
    """
    Same as `pow`, but raises `OverflowError` if the result gets too large.
    """
    operands = _collapse_operands(seq, eps)
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
    """
    Same as `*`, but if a or b is a string and the result gets too big, raises `MemoryError`.
    """
    for _ in range(2):
        if isinstance(a, str):
            if b <= 0:
                return ""
            if len(a) * abs(b) > 1e7:
                raise MemoryError("expression creates too large a string")
        a, b = b, a
    return a * b


def constrained_factorial(x):
    """
    Same as `math.factorial`, but raises `ValueError` if x is under 0 or over 32,767.
    """
    if not (0 <= x < 32768):
        raise ValueError(f"{x!r} not in working 0-32,767 range")
    if math.isclose(x, int(x), abs_tol=1e-12):
        x = int(round(x))
    return math.factorial(x)


@total_ordering
class ArithNode:
    def __init__(self, *args):
        self.tokens = args[-1][0]
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


class NullNode(ArithNode):
    def __init__(self, tokens=("",)):
        super().__init__(tokens)

    def evaluate(self):
        return None


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
        return f"{{{', '.join(map(repr, self.tokens))}}}" if self.tokens else "{}"


class UnaryNode(ArithNode, ABC):
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


class BinaryNode(ArithNode, ABC):
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
        return f"({' '.join(map(repr, repr_tokens))})"


class TernaryNode(ArithNode):
    opns_map: Dict[Tuple[str, str], Callable] = {}

    def left_associative_evaluate(self, oper_fn_map):
        operands = self.tokens
        ret = operands[0].evaluate()
        i = 1
        while i < len(operands):
            op1, operand1, op2, operand2 = operands[i : i + 4]
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
        return f"({' '.join(map(repr, repr_tokens))})"


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
    opns_map: Dict[str, Callable] = {}

    def evaluate(self):
        with _trimming_exception_traceback():
            return self.left_associative_evaluate(self.opns_map)


class ArithmeticBinaryOp(BinaryNode):
    opns_map: Dict[str, Callable] = {
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
            if not all(isinstance(op, _numeric_type) for op in operands):
                raise TypeError("invalid operators for exponentiation")

            return safe_pow(*operands)


class ArithmeticFunction(ArithNode):
    fn_map: Dict[str, FunctionSpec] = None

    def evaluate(self):
        with _trimming_exception_traceback():
            fn_name, *fn_args = self.tokens
            if fn_name not in self.fn_map:
                raise ValueError(f"{fn_name!r} is not a recognized function")
            fn_spec = self.fn_map[fn_name]

            if isinstance(fn_spec.arity, tuple):
                if not any(arit == len(fn_args) for arit in fn_spec.arity):
                    raise TypeError(
                        f"{fn_name} takes"
                        f" {' or '.join(str(ari) for ari in fn_spec.arity)}"
                        f" arg{'s' if fn_spec.arity[-1] != 1 else ''},"
                        f" {len(fn_args)} given"
                    )

            elif fn_spec.arity not in (len(fn_args), ...):
                raise TypeError(
                    f"{fn_name} takes"
                    f" {fn_spec.arity}"
                    f" arg{'s' if fn_spec.arity[-1] != 1 else ''},"
                    f" {len(fn_args)} given"
                )
            return fn_spec.method(*[arg.evaluate() for arg in fn_args])

    def __repr__(self):
        return f"{self.tokens[0]}({', '.join(map(repr, self.tokens[1:]))})"


class UnaryNot(UnaryNode):
    def evaluate(self):
        with _trimming_exception_traceback():
            return self.right_associative_evaluate({"not": operator.not_})


class BinaryLogicalOperator(BinaryNode):
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


class BinaryComparison(BinaryNode):
    epsilon = 1e-15

    opns_map = {
        "<": partial(_lt, eps=epsilon),
        "<=": partial(_le, eps=epsilon),
        ">": partial(_gt, eps=epsilon),
        ">=": partial(_ge, eps=epsilon),
        "==": partial(_eq, eps=epsilon),
        "!=": partial(_ne, eps=epsilon),
        "≠": partial(_ne, eps=epsilon),
        "≤": partial(_le, eps=epsilon),
        "≥": partial(_ge, eps=epsilon),
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


def make_in_container_node(ident_node_class, set_bin_op_class):
    def _inner(t):
        ret = InContainerNode(t)
        ret.identifier_node_class = ident_node_class
        ret.SetBinaryOp = set_bin_op_class
        return ret

    return _inner


class InContainerNode(UnaryNode):
    def evaluate(self):
        operand, op, range_expr = self.tokens
        assert_negate_fn = (lambda x: not not x, lambda x: not x)[op in ("∉", "not_in")]

        # noinspection PyUnresolvedReferences
        if isinstance(range_expr, (self.identifier_node_class, self.SetBinaryOp)):
            range_expr = range_expr.evaluate()

        with _trimming_exception_traceback():
            op_val = operand.evaluate()
            return assert_negate_fn(
                sum(op_val == elem for elem in range_expr.evaluate())
                if isinstance(range_expr, SetNode)
                else op_val in range_expr
            )

    def __repr__(self):
        operand, op, range_expr = self.tokens

        lower_symbol = "[" if range_expr.lower_inclusive else "("
        upper_symbol = "]" if range_expr.upper_inclusive else ")"
        repr_format = (f"{operand!r} {op}"
                       f" {lower_symbol}{range_expr.lower!r}, {range_expr.upper!r}{upper_symbol}")
        return repr_format


class RoundToEpsilon:
    def __init__(self, result):
        self._result = result
        self.epsilon = 1e-15

    def __repr__(self):
        return f"~{self._result}"

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


class AsDecimal:
    def __init__(self, result):
        self._result = result
        self.epsilon = 10**-decimal.getcontext().prec

    def __repr__(self):
        return f"{self._result}"

    def evaluate(self):
        with _trimming_exception_traceback():
            # print(self._result.dump())
            ret = self._result[0].evaluate()
            if isinstance(ret, (float, complex)):
                if math.isclose(ret.imag, 0, abs_tol=self.epsilon):
                    ret = decimal.Decimal(ret.real)
                if math.isclose(ret.real, 0, abs_tol=self.epsilon):
                    if ret.imag:
                        ret = complex(0, ret.imag)
                    else:
                        ret = 0
            return ret


DEFAULT_BASE_FUNCTION_MAP: Dict[str, FunctionSpec] = {
    "abs": FunctionSpec("abs", abs, 1),
    "round": FunctionSpec("round", round, (1, 2)),
    "trunc": FunctionSpec("trunc", math.trunc, 1),
    "ceil": FunctionSpec("ceil", math.ceil, 1),
    "floor": FunctionSpec("floor", math.floor, 1),
    "min": FunctionSpec("min", min, ...),
    "max": FunctionSpec("max", max, ...),
    "str": FunctionSpec("str", str, 1),
    "bool": FunctionSpec("bool", lambda x: not not x, 1),
}


class BaseArithmeticParser:
    """
    Base class for defining arithmetic parsers. Options are:

    max_vars: :class:`int`
        Represents the maximum number of variables that can be defined. Default to ``1000``.
    max_memory: :class:`int`
        Represents the maximum space in memory that can be allocated to variables and user functions.
        Default to ``10**6``.
    allow_user_variables: :class:`bool`
        Allows/disallows user defined variables. Default to ``True``.
    allow_user_functions: :class:`bool`
        Allows/disallows user defined variables. Default to the value of ``allow_user_variables``.
    base_function_map: :class:`dict`
        Base function map. Default to ``DEFAULT_BASE_FUNCTION_MAP``.
    epsilon: :class:`float`
        Represents error approximation. Default to ``1e-15``.
    max_expression_depth: :class:`int`
        Represents maximum expression depth. Default to ``6``.
    max_formula_depth: :class:`int`
        Represents maximum expression depth. Default to ``12``.
    max_set_depth: :class:`int`
        Represents maximum set depth. Default to ``6``.
    use_decimal: :class:`bool`
        Use decimal.Decimal values instead of float values. Default to ``False``.
    """

    LEFT = pp.opAssoc.LEFT
    RIGHT = pp.opAssoc.RIGHT

    def usage(self) -> str:
        import textwrap

        msg = textwrap.dedent(
            """\
        Enter an arithmetic expression or assignment statement, using
        the following operators:
        {operator_list}

        Multiple assignments can be made using lists of variable names and
        corresponding lists of expressions (lists must be of matching lengths).
            x₁, y₁ = 1, 2
            a, b, c = 1, 2, a+b

        {user_defined_functions}
        Expression can include the following functions:
        {function_list}
        """
        )
        func_list = _make_name_list_string(
            names=list({**self.base_function_map, **self.added_function_specs}),
            indent="  ",
        )
        custom_operators = [
            str(oper_defn[0]) for oper_defn in self.added_operator_specs
        ]
        operators = self._base_operators

        oper_list = _make_name_list_string(
            names=custom_operators + operators + ["|absolute-value|"], indent="  "
        )
        if self.user_defined_functions_supported:
            user_defined_functions = textwrap.dedent(
                """\
                Deferred evaluation assignments can be defined using "@=":
                    circle_area @= pi * r**2
                    circle_area
                will re-evaluate 'circle_area' using the current value of 'r'.
                """
            )
        else:
            user_defined_functions = ""

        return msg.format(
            function_list=func_list,
            operator_list=oper_list,
            user_defined_functions=user_defined_functions,
        )

    class IdentifierNode(ArithNode):
        _assigned_vars: Dict[str, Any] = {}

        @property
        def name(self):
            return self.tokens

        def evaluate(self):
            with _trimming_exception_traceback():
                if self.name in self._assigned_vars:
                    return self._assigned_vars[self.name].evaluate()
                else:
                    raise NameError(f"variable {self.name!r} not known")

        def __repr__(self):
            return self.name

    # noinspection PyUnresolvedReferences
    class Operators:
        # noinspection PyUnresolvedReferences
        _NOT_IN = (NOT() + IN()).addParseAction("_".join)

        EXPONENT = OperatorSpec("**", 2, pp.opAssoc.LEFT, ExponentBinaryOp)
        UNARY_SIGN = OperatorSpec(
            pp.oneOf("+ - −"), 1, pp.opAssoc.RIGHT, ArithmeticUnaryOp
        )
        MULTIPLICATION = OperatorSpec(
            pp.oneOf("* // / mod × ÷"), 2, pp.opAssoc.LEFT, ArithmeticBinaryOp
        )
        ADDITION = OperatorSpec(
            pp.oneOf("+ - −"), 2, pp.opAssoc.LEFT, ArithmeticBinaryOp
        )
        INEQUALITY = OperatorSpec(
            pp.oneOf("< > <= >= == != ≠ ≤ ≥"), 2, pp.opAssoc.LEFT, BinaryComparison
        )
        IS_ELEMENT_set_expression = pp.Forward()
        IS_ELEMENT_var_name = pp.Forward()
        IS_ELEMENT = OperatorSpec(
            (IN | _NOT_IN | pp.oneOf("∈ ∉"))
            - (IS_ELEMENT_set_expression | IS_ELEMENT_var_name),
            1,
            pp.opAssoc.LEFT,
            None,
        )

        LOGICAL_NOT = OperatorSpec(NOT, 1, pp.opAssoc.RIGHT, UnaryNot)
        LOGICAL_AND = OperatorSpec(AND | "∧", 2, pp.opAssoc.LEFT, BinaryLogicalOperator)
        LOGICAL_OR = OperatorSpec(OR | "∨", 2, pp.opAssoc.LEFT, BinaryLogicalOperator)
        C_STYLE_TERNARY = OperatorSpec(("?", ":"), 3, pp.opAssoc.RIGHT, TernaryComp)

    def __init__(self, **options):
        self.max_number_of_vars = options.get("max_vars", 1000)
        self.max_var_memory = options.get("max_memory", 10 ** 6)
        self.user_variables_supported = options.get("allow_user_variables", True)
        self.user_defined_functions_supported = options.get(
            "allow_user_functions", self.user_variables_supported
        )
        self.using_decimal = options.get("use_decimal", False)
        self.float_parser_class = AsDecimal if self.using_decimal else RoundToEpsilon

        self._added_operator_specs: List[OperatorSpec] = []
        self._added_function_specs: Dict[str, FunctionSpec] = {}
        self._base_operators: List[str] = (
            "** * // / mod × ÷ + - < > <= >= == != ≠ ≤ ≥ ∈ ∉ ∩ ∪ & | in not and ∧ or ∨ ?:"
        ).split()
        self._base_function_map: Dict[str, FunctionSpec] = options.get("base_function_map", DEFAULT_BASE_FUNCTION_MAP)

        # epsilon for computing "close" floating point values - can be updated in customize
        self.epsilon = options.get("epsilon", 1e-15)

        # customize can update or replace with different characters
        self.ident_letters = (
            pp.srange("[A-Za-z]") + pp.srange("[ªºÀ-ÖØ-öø-ÿ]") + pp.srange("[Α-Ωα-ω]")
        )

        # customize can raise or lower the maximum expression depth
        # to be supported - default = 6
        self.maximum_expression_depth = options.get("max_expression_depth", 6)
        self.maximum_formula_depth = options.get("max_formula_depth", 12)
        self.maximum_set_depth = options.get("max_set_depth", 6)

        # storage for assigned variables
        self._variable_map = {}

        # customize can add pre-defined constants
        self._initial_variables = {}

        self.customize()
        self._parser = self.make_parser()
        self.parse = self._parse

    @property
    def base_function_map(self) -> Dict[str, FunctionSpec]:
        return {**self._base_function_map}

    @property
    def added_function_specs(self) -> Dict[str, FunctionSpec]:
        return {**self._added_function_specs}

    @property
    def added_operator_specs(self) -> List[OperatorSpec]:
        return self._added_operator_specs[:]

    def scan_string(self, *args):
        yield from self.get_parser().scanString(*args)

    def scanString(self, *args):
        yield from self.scan_string(*args)

    def _parse(self, *args, **kwargs):
        """Parses an expression."""
        if _get_expression_depth(args[0]) > self.maximum_expression_depth:
            raise OverflowError("expression too deeply nested")

        if _get_set_depth(args[0]) > self.maximum_set_depth:
            raise OverflowError("set too deeply nested")

        parsed = self.get_parser().parseString(*args, **kwargs)

        if parsed:
            return parsed[0]

    def evaluate(self, arith_expression):
        """
        Evaluates an expression and returns its result.
        """
        with _trimming_exception_traceback():
            parsed = self.parse(arith_expression, parseAll=True)
            return parsed.evaluate()

    def __getattr__(self, attr):
        parser = self._parser
        if hasattr(parser, attr):
            return getattr(parser, attr)
        raise AttributeError(f"no such attribute {attr!r}")

    # define methods for item getting, to access variable values created
    # in the parser
    def __getitem__(self, key):
        self_vars = self.vars()
        if key in self_vars:
            return self_vars[key]
        else:
            raise NameError(f"no such variable {key!r}")

    # make class not iterable, which happens by default when __getitem__ is defined
    __iter__ = None

    def __delitem__(self, key):
        self_vars = self._variable_map
        if key in self_vars:
            del self_vars[key]

    def __setitem__(self, name, value):
        self._variable_map[name] = LiteralNode([value])

    def customize(self) -> None:
        """
        Entry point to define operators, functions and variables.
        """

    def add_operator(self,
                     operator_expr: Union[str, pp.ParserElement],
                     arity: int,
                     assoc: object,
                     parse_action: Callable) -> None:
        """
        Adds an operator.

        Parameters:
        - operator_expr (str): operator expression
        - arity (int): operator arity
        - assoc: `ArithmeticParser.RIGHT` or `ArithmeticParser.LEFT`
        - parse_action: method to associate with the operator
        """
        operator_node_superclass = {
            (1, pp.opAssoc.LEFT): ArithmeticUnaryPostOp,
            (1, pp.opAssoc.RIGHT): ArithmeticUnaryOp,
            (2, pp.opAssoc.LEFT): ArithmeticBinaryOp,
            (2, pp.opAssoc.RIGHT): ArithmeticBinaryOp,
            (3, pp.opAssoc.LEFT): TernaryNode,
            (3, pp.opAssoc.RIGHT): TernaryNode,
        }[arity, assoc]
        if isinstance(parse_action, dict):
            operator_node_class = type(
                "",
                (operator_node_superclass,),
                {"opns_map": parse_action},
            )
        else:
            operator_node_class = type(
                "",
                (operator_node_superclass,),
                {"opns_map": {str(operator_expr): parse_action}},
            )
        if isinstance(operator_expr, str) and operator_expr.isalpha():
            operator_expr = pp.Keyword(operator_expr, identChars=string.ascii_letters)
        self._added_operator_specs.insert(
            0, OperatorSpec(operator_expr, arity, assoc, operator_node_class)
        )

    def initialize_variable(self, vname: str, vvalue: Any, as_formula: bool = False) -> None:
        """
        Adds a variable to the parser.

        Parameters:
        - vname (str): variable name
        - vvalue: variable value
        - as_formula (bool): if True, variable is registered as a formula
          and its value can be dynamically updated (defaults to False)
        """
        self._initial_variables[vname] = (vvalue, as_formula)

    def add_function(self, fn_name: str, fn_arity: Union[int, Tuple[int, int]], fn_method: Callable):
        """
        Adds a function to the parser.

        Parameters:
        - fn_name (str): the name of the function
        - fn_arity (tuple, int or ...): number of arguments accepted by the function
        - fn_method: the method associated with the function
        """
        self._added_function_specs[fn_name] = FunctionSpec(fn_name, fn_method, fn_arity)

    def get_parser(self) -> pp.ParserElement:
        """
        Retrieves parser or make a new one if None was found.
        """
        if self._parser is None:
            self._parser = self.make_parser()
        return self._parser

    def make_parser(self) -> pp.ParserElement:
        """
        Creates a new parser.
        """
        arith_operand = pp.Forward()
        LPAR, RPAR, LBRACK, RBRACK, LBRACE, RBRACE, COMMA = map(pp.Suppress, "()[]{},")
        # superscripts = "⁻¹²³⁴⁵⁶⁷⁸⁹"
        fn_name_expr = pp.Combine(
            pp.Word(
                "_" + self.ident_letters, "_" + self.ident_letters + pp.nums
            ) + pp.Optional(pp.oneOf("⁻¹ ² ³"))
        )
        function_expression = pp.Group(
            fn_name_expr("fn_name")
            + LPAR
            + pp.Optional(pp.delimitedList(arith_operand))("args")
            + RPAR
        )
        function_node_class = type(
            "Function",
            (ArithmeticFunction,),
            {"fn_map": {**self._base_function_map, **self._added_function_specs}},
        )
        function_expression.addParseAction(function_node_class)

        set_operand = pp.Group(
            LBRACE + pp.Optional(pp.delimitedList(arith_operand)) + RBRACE
        ).addParseAction(SetNode)

        if self.using_decimal:
            number = (
                    ppc.real.set_parse_action(pp.token_map(decimal.Decimal))
                    | ppc.integer.set_parse_action(pp.token_map(decimal.Decimal))
            )
            numeric_operand = number.addParseAction(LiteralNode)
        else:
            numeric_operand = ppc.number().addParseAction(LiteralNode)
        qs = pp.QuotedString('"', escChar="\\") | pp.QuotedString("'", escChar="\\")
        string_operand = qs.addParseAction(LiteralNode)
        # noinspection PyUnresolvedReferences
        bool_operand = (TRUE | FALSE).addParseAction(LiteralNode)

        ident_sub_chars = pp.srange("[ₐ-ₜ]") + pp.srange("[₀-₉]")
        var_name = pp.Combine(
            ~any_keyword
            + pp.Word("_" + self.ident_letters, "_" + self.ident_letters + pp.nums)
            + pp.Optional(pp.Word(ident_sub_chars))
        ).setName("identifier")

        identifier_node_class = type(
            "Identifier", (self.IdentifierNode,), {"_assigned_vars": self._variable_map}
        )
        var_name.addParseAction(identifier_node_class)

        # fmt: off
        def make_sets(a, b):
            a_set = a if isinstance(a, PrettySet) else PrettySet(elem.evaluate() for elem in a)
            b_set = b if isinstance(b, PrettySet) else PrettySet(elem.evaluate() for elem in b)
            return a_set, b_set
        # fmt: on

        def set_intersection(a, b):
            """
            Represents a set intersection.
            """
            a_set, b_set = make_sets(a, b)
            return PrettySet(a_set.intersection(b_set))

        def set_union(a, b):
            """
            Represents a set union.
            """
            a_set, b_set = make_sets(a, b)
            return PrettySet(a_set.union(b_set))

        def set_difference(a, b):
            """
            Represents a set difference.
            """
            a_set, b_set = make_sets(a, b)
            return PrettySet(a_set.difference(b_set))

        def set_symmetric_difference(a, b):
            """
            Represents a set symmetric difference.
            """
            a_set, b_set = make_sets(a, b)
            return PrettySet(a_set.symmetric_difference(b_set))

        class SetBinaryOp(BinaryNode):
            opns_map = {
                "∩": set_intersection,
                "&": set_intersection,
                "∪": set_union,
                "|": set_union,
                "-": set_difference,
                "−": set_difference,
                "^": set_symmetric_difference,
                "∆": set_symmetric_difference,
            }

            def evaluate(self):
                with _trimming_exception_traceback():
                    return self.left_associative_evaluate(self.opns_map)

        set_expression = pp.infixNotation(
            set_operand | var_name,
            [
                (pp.oneOf("∩ & ∪ | - − ^ ∆"), 2, pp.opAssoc.LEFT, SetBinaryOp),
            ],
        )

        BaseArithmeticParser.Operators.IS_ELEMENT_set_expression <<= set_expression
        BaseArithmeticParser.Operators.IS_ELEMENT_var_name <<= var_name

        base_operator_specs = [
            BaseArithmeticParser.Operators.EXPONENT,
            BaseArithmeticParser.Operators.UNARY_SIGN,
            BaseArithmeticParser.Operators.MULTIPLICATION,
            BaseArithmeticParser.Operators.ADDITION,
            BaseArithmeticParser.Operators.INEQUALITY,
            BaseArithmeticParser.Operators.IS_ELEMENT._replace(
                action=make_in_container_node(identifier_node_class, SetBinaryOp)
            ),
            BaseArithmeticParser.Operators.LOGICAL_NOT,
            BaseArithmeticParser.Operators.LOGICAL_AND,
            BaseArithmeticParser.Operators.LOGICAL_OR,
            BaseArithmeticParser.Operators.C_STYLE_TERNARY,
        ]

        ABS_VALUE_VERT = pp.Suppress("|")
        abs_value_expression = ABS_VALUE_VERT + arith_operand + ABS_VALUE_VERT

        def cvt_to_function_call(function):
            def wrapper(tokens):
                ret = pp.ParseResults([function]) + tokens
                ret["fn_name"] = function
                ret["args"] = tokens
                return [ret]

            return wrapper

        abs_value_expression.addParseAction(
            cvt_to_function_call("abs"), function_node_class
        )

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
            + pp.oneOf("<- = ←")
            + pp.delimitedList(rvalue)("rhs")
        )

        value_clear_statement = (
            pp.delimitedList(lvalue)("lhs") + pp.oneOf("<- = ←") + pp.StringEnd()
        )
        value_clear_statement.setName("value clear statement")

        def eval_and_store_value(tokens):
            if not self.user_variables_supported:
                raise Exception("user variable definitions are not allowed!")
            if len(tokens.lhs) > len(tokens.rhs):
                raise TypeError("not enough values given")
            if len(tokens.lhs) < len(tokens.rhs):
                raise TypeError("not enough variable names given")

            assignments = []
            # noinspection PyUnresolvedReferences
            assigned_vars = identifier_node_class._assigned_vars
            rval = None
            for lhs_name, rhs_expr in zip(tokens.lhs, tokens.rhs):
                rval = LiteralNode([rhs_expr.evaluate()])
                dest_var_name = lhs_name.name
                if (
                    dest_var_name not in assigned_vars
                    and len(assigned_vars) >= self.max_number_of_vars
                ):
                    raise Exception("too many variables defined")
                assigned_vars[dest_var_name] = rval
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
            if not self.user_defined_functions_supported:
                raise Exception("user function definitions are not allowed!")
            # see if this var_name is recursively referenced
            formula_defn = tokens.rhs
            dest_var_name = tokens.lhs.name

            to_visit = deque([formula_defn])
            while to_visit:
                cur_expr = to_visit.popleft()
                if isinstance(cur_expr, self.IdentifierNode):
                    if cur_expr.name == dest_var_name:
                        raise Exception(
                            f"illegal recursion, {dest_var_name!r} is used in expression"
                        )

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
            # noinspection PyUnresolvedReferences
            assigned_vars = identifier_node_class._assigned_vars
            if (
                dest_var_name not in assigned_vars
                and len(assigned_vars) >= self.max_number_of_vars
                or sum(sys.getsizeof(vv) for vv in assigned_vars.values())
                > self.max_var_memory
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
            # noinspection PyUnresolvedReferences
            assigned_vars = identifier_node_class._assigned_vars
            for var in tokens.lhs:
                assigned_vars.pop(var.name, None)
            return [NullNode()]

        value_assignment_statement.addParseAction(RoundToEpsilon)
        lone_rvalue = rvalue().addParseAction(RoundToEpsilon)
        formula_assignment_statement.addParseAction(verify_formula_not_recursive)
        formula_assignment_statement.addParseAction(store_parsed_value)
        value_clear_statement.addParseAction(clear_parsed_value)

        # fmt: off
        parser = (
                value_assignment_statement
                | value_clear_statement
                | formula_assignment_statement
                | lone_rvalue
        )
        # fmt: on

        # init _variable_map with any pre-defined values
        for varname, (varvalue, as_formula) in self._initial_variables.items():
            if as_formula:
                try:
                    parser.parseString(
                        f"{varname} {formula_assignment_operator} {varvalue}"
                    )
                except NameError:
                    pass
            else:
                self._variable_map[varname] = LiteralNode([varvalue])
        return parser

    def vars(self):
        """Returns all the variables defined in the parser."""
        ret = {}
        for k, v in self._variable_map.items():
            if isinstance(v, LiteralNode):
                ret[k] = v.evaluate()
            else:
                ret[k] = repr(v)
        return ret

    def __getstate__(self):
        return (
            self._parser,
            self._variable_map,
            self._initial_variables,
            self.maximum_formula_depth,
            self.maximum_expression_depth,
            self.maximum_set_depth,
            self.max_number_of_vars,
            self.user_defined_functions_supported,
            self.user_variables_supported,
            self._added_function_specs,
            self._added_operator_specs,
            self._base_operators,
            self._base_function_map,
            self.epsilon,
            self.ident_letters,
            self.using_decimal,
        )

    def __setstate__(self, state):
        (
            self._parser,
            self._variable_map,
            self._initial_variables,
            self.maximum_formula_depth,
            self.maximum_expression_depth,
            self.maximum_set_depth,
            self.max_number_of_vars,
            self.user_defined_functions_supported,
            self.user_variables_supported,
            self._added_function_specs,
            self._added_operator_specs,
            self._base_operators,
            self._base_function_map,
            self.epsilon,
            self.ident_letters,
            self.using_decimal,
        ) = state
        self.parse = self._parse


def log(x: Union[int, float], y: Union[int, float] = math.e) -> float:
    """
    Similar to `math.log`, with is_close for bases 2 and 10.
    """
    if math.isclose(y, 2, abs_tol=1e-15):
        return math.log2(x)
    if math.isclose(y, 10, abs_tol=1e-15):
        return math.log10(x)
    return math.log(x, y)


class ArithmeticParser(BaseArithmeticParser):
    """
    Ready to use, basic parser. Example:
    ```python
    >>> parser = ArithmeticParser()
    >>> parser.evaluate("gamma(2)") # Predefined functions
    1
    >>> parser.evaluate("pi*9") # Predefined variables
    28.274333882308138
    >>> parser.evaluate("9**3")
    729
    ```
    """

    class Operators:
        """
        Custom operator definitions, beyond those defined in ArithmeticParser.
        """

        # avoid clash with '!=' operator
        _factorial_operator = (~pp.Literal("!=") + "!").setName("!")
        FACTORIAL = OperatorSpec(
            _factorial_operator, 1, pp.opAssoc.LEFT, constrained_factorial
        )

        radical_sign = pp.Literal("√")
        root_integer = pp.oneOf("² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹").leaveWhitespace()
        radical_operator_expr = pp.Combine(
            pp.Optional(root_integer, default="²") + radical_sign
        )
        radical_operator_expr.setName("ⁿ√")
        unary_root_methods_map = {
            "²√": (lambda x: x ** (1 / 2)),
            "³√": (lambda x: x ** (1 / 3)),
            "⁴√": (lambda x: x ** (1 / 4)),
            "⁵√": (lambda x: x ** (1 / 5)),
            "⁶√": (lambda x: x ** (1 / 6)),
            "⁷√": (lambda x: x ** (1 / 7)),
            "⁸√": (lambda x: x ** (1 / 8)),
            "⁹√": (lambda x: x ** (1 / 9)),
        }
        RADICAL_UNARY = OperatorSpec(
            radical_operator_expr, 1, pp.opAssoc.RIGHT, unary_root_methods_map
        )
        binary_root_methods_map = {
            "²√": (lambda x, y: x * y ** (1 / 2)),
            "³√": (lambda x, y: x * y ** (1 / 3)),
            "⁴√": (lambda x, y: x * y ** (1 / 4)),
            "⁵√": (lambda x, y: x * y ** (1 / 5)),
            "⁶√": (lambda x, y: x * y ** (1 / 6)),
            "⁷√": (lambda x, y: x * y ** (1 / 7)),
            "⁸√": (lambda x, y: x * y ** (1 / 8)),
            "⁹√": (lambda x, y: x * y ** (1 / 9)),
        }
        RADICAL_BINARY = OperatorSpec(
            radical_operator_expr, 2, pp.opAssoc.LEFT, binary_root_methods_map
        )

        DEGREE_OPERATOR = OperatorSpec("°", 1, pp.opAssoc.LEFT, math.radians)

        special_exponents_methods_map = {
            "⁻¹": (lambda x: 1 / x),
            "⁰": (lambda x: x ** 0),
            "¹": (lambda x: x),
            "²": (lambda x: safe_pow(x, 2)),
            "³": (lambda x: safe_pow(x, 3)),
        }
        SPECIAL_EXPONENTS = OperatorSpec(
            pp.oneOf("⁻¹ ⁰ ¹ ² ³").leaveWhitespace(),
            1,
            pp.opAssoc.LEFT,
            special_exponents_methods_map,
        )

    def customize(self):
        """
        Entry point to define operators, functions and variables.
        """
        import math

        super().customize()
        phi = (1.0 + 5 ** 0.5) / 2.0  # The golden number

        self.add_operator(*ArithmeticParser.Operators.RADICAL_UNARY)
        self.add_operator(*ArithmeticParser.Operators.SPECIAL_EXPONENTS)
        self.add_operator(*ArithmeticParser.Operators.RADICAL_BINARY)
        self.add_operator(*ArithmeticParser.Operators.DEGREE_OPERATOR)
        self.add_operator(*ArithmeticParser.Operators.FACTORIAL)

        def fn_raised_to(fn, n):
            return lambda x: fn(x) ** n

        self.add_function("sin", 1, math.sin)
        self.add_function("cos", 1, math.cos)
        self.add_function("tan", 1, math.tan)
        self.add_function("sin²", 1, fn_raised_to(math.sin, 2))
        self.add_function("cos²", 1, fn_raised_to(math.cos, 2))
        self.add_function("tan²", 1, fn_raised_to(math.tan, 2))
        self.add_function("sin³", 1, fn_raised_to(math.sin, 3))
        self.add_function("cos³", 1, fn_raised_to(math.cos, 3))
        self.add_function("tan³", 1, fn_raised_to(math.tan, 3))
        self.add_function("sin⁻¹", 1, math.asin)
        self.add_function("cos⁻¹", 1, math.acos)
        self.add_function("tan⁻¹", 1, math.atan)
        self.add_function("asin", 1, math.asin)
        self.add_function("acos", 1, math.acos)
        self.add_function("atan", 1, math.atan)
        self.add_function("sinh", 1, math.sinh)
        self.add_function("cosh", 1, math.cosh)
        self.add_function("tanh", 1, math.tanh)
        self.add_function("sinh²", 1, fn_raised_to(math.sinh, 2))
        self.add_function("cosh²", 1, fn_raised_to(math.cosh, 2))
        self.add_function("tanh²", 1, fn_raised_to(math.tanh, 2))
        self.add_function("sinh³", 1, fn_raised_to(math.sinh, 3))
        self.add_function("cosh³", 1, fn_raised_to(math.cosh, 3))
        self.add_function("tanh³", 1, fn_raised_to(math.tanh, 3))
        self.add_function("rad", 1, math.radians)
        self.add_function("deg", 1, math.degrees)
        self.add_function("ln", 1, math.log)
        self.add_function(
            "log", (1, 2), log
        )  # log function can accept one or two values
        self.add_function("log2", 1, math.log2)
        self.add_function("log10", 1, math.log10)
        self.add_function("gcd", 2, math.gcd)
        self.add_function(
            "lcm", 2, lambda a, b: abs(a * b) // math.gcd(a, b) if a or b else 0
        )
        self.add_function("gamma", 1, math.gamma)
        self.add_function(
            "hypot", ..., lambda *seq: sum(safe_pow(i, 2) for i in seq) ** 0.5
        )
        self.add_function("rnd", 0, random.random)
        self.add_function("randint", 2, random.randint)
        self.add_function(
            "sgn", 1, lambda x: 0 if _eq(x, 0, self.epsilon) else 1 if x > 0 else -1
        ),

        self.initialize_variable("pi", math.pi)
        self.initialize_variable("π", math.pi)
        self.initialize_variable("τ", math.tau)
        self.initialize_variable("tau", math.tau)
        self.initialize_variable("e", math.e)
        self.initialize_variable("φ", phi)
        self.initialize_variable("ϕ", phi)
        self.initialize_variable("phi", phi)
