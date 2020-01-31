#
# plusminus.py
#
"""
plusminus

plusminus is a module that builds on the pyparsing infixNotation helper method to build easy-to-code and easy-to-use
parsers for parsing and evaluating infix arithmetic expressions. arithmetic_parsing's ArithmeticParser class includes
separate parse and evaluate methods, handling operator precedence, override with parentheses, presence or absence of
whitespace, built-in functions, and pre-defined and user-defined variables.

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

from collections import namedtuple
from functools import partial
import math
import operator
import random
import pyparsing as pp

ppc = pp.pyparsing_common
pp.ParserElement.enablePackrat()

__all__ = "ArithmeticParser BasicArithmeticParser expressions any_keyword __version__".split()
__version__ = "0.1"

expressions = {}

# keywords
keywords = {
    k.upper(): pp.Keyword(k)
    for k in """between within from to in range and or not True False if else mod""".split()
}
vars().update(keywords)
expressions.update(keywords)

any_keyword = pp.MatchFirst(keywords.values())
IN_RANGE_FROM = (IN + RANGE + FROM).addParseAction('_'.join)
TRUE.addParseAction(lambda: True)
FALSE.addParseAction(lambda: False)

FunctionSpec = namedtuple("FunctionSpec", "method arity")

_numeric_type = (int, float, complex)

# define special versions of lt, le, etc. to comprehend "is close"
_lt = lambda a, b, eps: a < b and not math.isclose(a, b, abs_tol=eps) if isinstance(a, _numeric_type) and isinstance(b, _numeric_type) else a < b
_le = lambda a, b, eps: a <= b or math.isclose(a, b, abs_tol=eps) if isinstance(a, _numeric_type) and isinstance(b, _numeric_type) else a <= b
_gt = lambda a, b, eps: a > b and not math.isclose(a, b, abs_tol=eps) if isinstance(a, _numeric_type) and isinstance(b, _numeric_type) else a > b
_ge = lambda a, b, eps: a >= b or math.isclose(a, b, abs_tol=eps) if isinstance(a, _numeric_type) and isinstance(b, _numeric_type) else a >= b
_eq = lambda a, b, eps: a == b or math.isclose(a, b, abs_tol=eps) if isinstance(a, _numeric_type) and isinstance(b, _numeric_type) else a == b
_ne = lambda a, b, eps: not math.isclose(a, b, abs_tol=eps) if isinstance(a, _numeric_type) and isinstance(b, _numeric_type) else a != b


def collapse_operands(seq, eps=1e-15):
    cur = seq[:]
    # if any((a == 0 and b < 0) for a, b in zip(seq, seq[1:])):
    #     1 / 0
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
                if cur[i+1] < 0 and (i == len(cur)-2 or cur[i+2] % 2 != 0):
                    0 ** cur[i+1]
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


def safe_pow(seq, eps=1e-15):
    # print(seq)
    operands = collapse_operands(seq, eps)
    # print(operands)
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
            if 0 not in (ret, op1) and math.log10(abs(op1)) + math.log10(abs(ret)) > 8:
                raise OverflowError("operands too large for expression")
            ret = op1 ** ret
    return ret


def safe_mult(a, b):
    for _ in range(2):
        if isinstance(a, str):
            if b <= 0:
                return ''
            if  len(a) * abs(b) > 1e7:
                raise MemoryError("expression creates too large a string")
        a, b = b, a
    return a * b


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
        raise NotImplemented

    def right_associative_evaluate(self, oper_fn_map):
        pass

    def left_associative_evaluate(self, oper_fn_map):
        pass

    def __repr__(self):
        return type(self).__name__ + '/' + (", ".join(repr(t) for t in self.tokens)
                if self.iterable_tokens else repr(self.tokens))


class LiteralNode(ArithNode):
    def evaluate(self):
        return self.tokens

    def __repr__(self):
        return repr(self.tokens)


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
        return ''.join(map(repr, self.tokens))

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
        return "({})".format(''.join(map(repr, self.tokens)))

class TernaryNode(ArithNode):
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
        return self.left_associative_evaluate(self.opns_map)

    def __repr__(self):
        return "({})".format(''.join(map(repr, self.tokens)))


class ArithmeticParser:
    """
    Base class for defining arithmetic parsers.
    """
    LEFT = pp.opAssoc.LEFT
    RIGHT = pp.opAssoc.RIGHT
    MAX_VARS = 1000
    MAX_VAR_MEMORY = 10**6

    class ArithmeticUnaryOp(UnaryNode):
        opns_map = {
            '+': lambda x: x,
            '-': operator.neg,
            '−': operator.neg,
        }

        def evaluate(self):
            return self.right_associative_evaluate(self.opns_map)

    class ArithmeticUnaryPostOp(UnaryNode):
        opns_map = {}
        def evaluate(self):
            return self.left_associative_evaluate(self.opns_map)

    class ArithmeticBinaryOp(BinaryNode):
        opns_map = {
            '+': operator.add,
            '-': operator.sub,
            '−': operator.sub,
            '*': safe_mult,
            '/': operator.truediv,
            'mod': operator.mod,
            '×': safe_mult,
            '÷': operator.truediv,
        }

        def evaluate(self):
            return self.left_associative_evaluate(self.opns_map)

    class ExponentBinaryOp(ArithmeticBinaryOp):

        def evaluate(self):
            # parsed left-to-right, but evaluate right-to-left
            operands = [t.evaluate() for t in self.tokens[::2]]
            if not all(isinstance(op, (int, float, complex)) for op in operands):
                raise TypeError("invalid operators for exponentiation")

            return safe_pow(operands)

        def evaluateX(self):
            # parsed left-to-right, but evaluate right-to-left
            ret = self.tokens[-1].evaluate()
            for operand in self.tokens[-3::-2]:
                op1 = operand.evaluate()
                # rough guard against too large values in expression
                if math.log10(abs(op1)) + math.log10(abs(ret)) > 8:
                    raise OverflowError("operands too large for expression")
                ret = op1 ** ret
            return ret

    class IdentifierNode(ArithNode):
        _assigned_vars = {}

        @property
        def name(self):
            return self.tokens

        def evaluate(self):
            if self.name in self._assigned_vars:
                return self._assigned_vars[self.name].evaluate()
            else:
                raise NameError("variable {!r} not known".format(self.name))

        def __repr__(self):
            return self.name

    class ArithmeticFunction(ArithNode):
        def evaluate(self):
            fn_name, *fn_args = self.tokens
            if fn_name not in self.fn_map:
                raise ValueError("{!r} is not a recognized function".format(fn_name))
            fn_spec = self.fn_map[fn_name]
            if fn_spec.arity != len(fn_args):
                raise TypeError("{} takes {} arg{}, {} given".format(fn_name,
                                                                     fn_spec.arity,
                                                                     ('', 's')[fn_spec.arity != 1],
                                                                     len(fn_args)))
            return fn_spec.method(*[arg.evaluate() for arg in fn_args])

        def __repr__(self):
            return "{}({})".format(self.tokens[0], ','.join(map(repr, self.tokens[1:])))

    def __init__(self):
        self._added_operator_specs = []
        self._added_function_specs = {}
        self._base_operators = ("** * / mod × ÷ + - < > <= >= == != ≠ ≤ ≥ between-and within-and"
                               " in-range-from-to not and ∧ or ∨ ?:").split()
        self._base_function_map = {
            'sin': FunctionSpec(math.sin, 1),
            'cos': FunctionSpec(math.cos, 1),
            'tan': FunctionSpec(math.tan, 1),
            'asin': FunctionSpec(math.asin, 1),
            'acos': FunctionSpec(math.acos, 1),
            'atan': FunctionSpec(math.atan, 1),
            'sinh': FunctionSpec(math.sinh, 1),
            'cosh': FunctionSpec(math.cosh, 1),
            'tanh': FunctionSpec(math.tanh, 1),
            'rad': FunctionSpec(math.radians, 1),
            'deg': FunctionSpec(math.degrees, 1),
            'sgn': FunctionSpec((lambda x: -1 if x < 0 else 1 if x > 0 else 0), 1),
            'abs': FunctionSpec(abs, 1),
            'round': FunctionSpec(round, 2),
            'trunc': FunctionSpec(math.trunc, 1),
            'ceil': FunctionSpec(math.ceil, 1),
            'floor': FunctionSpec(math.floor, 1),
            'ln': FunctionSpec(math.log, 1),
            'log2': FunctionSpec(math.log2, 1),
            'log10': FunctionSpec(math.log10, 1),
            'gcd': FunctionSpec(math.gcd, 2),
            'lcm': FunctionSpec((lambda a, b: int(abs(a) / math.gcd(a, b) * abs(b)) if a or b else 0), 2),
            'gamma': FunctionSpec(math.gamma, 2),
            'hypot': FunctionSpec(math.hypot, 2),
        }

        # epsilon for computing "close" floating point values - can be updated in customize
        self.epsilon = 1e-15

        # customize can update or replace with different characters
        self.ident_letters = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzªº"
                              "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
                              + pp.srange("[Α-Ω]") + pp.srange("[α-ω]"))

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
        return self.get_parser().parseString(*args, **kwargs)[0]

    def evaluate(self, arith_expression):
        try:
            parsed = self.parse(arith_expression, parseAll=True)
            return parsed.evaluate()
        except Exception as e:
            raise e.with_traceback(None)

    def __getattr__(self, attr):
        parser = self._parser
        if hasattr(parser, attr):
            return getattr(parser, attr)
        raise AttributeError("no such attribute {!r}".format(attr))

    def __getitem__(self, key):
        return self.vars()[key]

    def __iter__(self):
        raise NotImplemented

    def customize(self):
        pass

    def add_operator(self, operator_expr, arity, assoc, parse_action):
        if isinstance(operator_expr, str) and callable(parse_action):
            operator_node_superclass = {
                (1, pp.opAssoc.LEFT): self.ArithmeticUnaryPostOp,
                (1, pp.opAssoc.RIGHT): self.ArithmeticUnaryOp,
                (2, pp.opAssoc.LEFT): self.ArithmeticBinaryOp,
                (2, pp.opAssoc.RIGHT): self.ArithmeticBinaryOp,
                (3, pp.opAssoc.LEFT): TernaryNode,
                (3, pp.opAssoc.RIGHT): TernaryNode,
            }[arity, assoc]
            operator_node_class = type('', (operator_node_superclass,),
                                       {'opns_map': {operator_expr: parse_action}})
        else:
            operator_node_class = parse_action
        self._added_operator_specs.insert(0, (operator_expr, arity, assoc, operator_node_class))

    def initialize_variable(self, vname, vvalue, as_formula=False):
        self._initial_variables[vname] = (vvalue, as_formula)

    def add_function(self, fn_name, fn_arity, fn_method):
        self._added_function_specs[fn_name] = (FunctionSpec(fn_method, fn_arity))

    def get_parser(self):
        if self._parser is None:
            self._parser = self.make_parser()
        return self._parser

    def make_parser(self):
        arith_operand = pp.Forward()
        LPAR, RPAR = map(pp.Suppress, "()")
        fn_name_expr = pp.Word('_' + self.ident_letters, '_' + self.ident_letters + pp.nums)
        function_expression = pp.Group(fn_name_expr("fn_name")
                                       + LPAR
                                       + pp.Optional(pp.delimitedList(arith_operand))("args")
                                       + RPAR)
        function_node_class = type("Function", (self.ArithmeticFunction,),
                                   {'fn_map': {**self._base_function_map, **self._added_function_specs}})
        function_expression.addParseAction(function_node_class)

        numeric_operand = ppc.number().addParseAction(LiteralNode)
        qs = pp.QuotedString('"', escChar="\\") | pp.QuotedString("'", escChar="\\")
        string_operand = qs.addParseAction(LiteralNode)
        bool_operand = (TRUE | FALSE).addParseAction(LiteralNode)

        ident_sub_chars = pp.srange("[ₐ-ₜ]") + pp.srange("[₀-₉]")
        var_name = pp.Combine(~any_keyword
                              + pp.Word('_' + self.ident_letters, '_' + self.ident_letters + pp.nums)
                              + pp.Optional(pp.Word(ident_sub_chars))
                              ).setName("identifier")

        class BinaryComparison(BinaryNode):
            def __init__(self, tokens):
                super().__init__(tokens)
                self.epsilon = 1e-15

            opns_map = {
                '<': partial(_lt, eps=self.epsilon),
                '<=': partial(_le, eps=self.epsilon),
                '>': partial(_gt, eps=self.epsilon),
                '>=': partial(_ge, eps=self.epsilon),
                '==': partial(_eq, eps=self.epsilon),
                '!=': partial(_ne, eps=self.epsilon),
                '≠': partial(_ne, eps=self.epsilon),
                '≤': partial(_le, eps=self.epsilon),
                '≥': partial(_ge, eps=self.epsilon),
            }

            def evaluate(self):
                return self.left_associative_evaluate(self.opns_map)

            def left_associative_evaluate(self, oper_fn_map):
                last = self.tokens[0].evaluate()
                ret = True
                for oper, operand in zip(self.tokens[1::2], self.tokens[2::2]):
                    next_ = operand.evaluate()
                    ret = ret and oper_fn_map[oper](last, next_)
                    last = next_
                return ret

        class IntervalComparison(TernaryNode):
            def __init__(self, tokens):
                super().__init__(tokens)
                self.epsilon = 1e-15

            opns_map = {
                ('between', 'and'): (lambda a, b, c: _lt(b, a, eps=self.epsilon) and _lt(a, c, eps=self.epsilon)),
                ('within', 'and'): (lambda a, b, c: _le(b, a, eps=self.epsilon) and _le(a, c, eps=self.epsilon)),
                ('in_range_from', 'to'): (lambda a, b, c: _le(b, a, eps=self.epsilon) and _lt(a, c, eps=self.epsilon)),
            }

        class UnaryNot(UnaryNode):
            def evaluate(self):
                return self.right_associative_evaluate({'not': operator.not_})

        class BinaryComp(BinaryNode):
            opns_map = {
                'and': operator.and_,
                'or': operator.or_,
                '∧': operator.and_,
                '∨': operator.or_,
            }

            def evaluate(self):
                return self.left_associative_evaluate(self.opns_map)

            def left_associative_evaluate(self, oper_fn_map):
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
                ('?', ':'): (lambda a, b, c: b if a else c),
            }

        class RoundToEpsilon:
            def __init__(self, result):
                self._result = result
                self.epsilon = 1e-15

            def evaluate(self):
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
                    if (not isinstance(ret, complex)
                            and abs(ret) < 1e15
                            and math.isclose(ret, int(ret), abs_tol=self.epsilon)):
                        return int(ret)
                return ret

        identifier_node_class = type('Identifier', (self.IdentifierNode,), {'_assigned_vars': self._variable_map})
        var_name.addParseAction(identifier_node_class)
        base_operator_specs = [
            ('**', 2, pp.opAssoc.LEFT, self.ExponentBinaryOp),
            ('-', 1, pp.opAssoc.RIGHT, self.ArithmeticUnaryOp),
            (pp.oneOf('* / mod × ÷'), 2, pp.opAssoc.LEFT, self.ArithmeticBinaryOp),
            (pp.oneOf('+ - −'), 2, pp.opAssoc.LEFT, self.ArithmeticBinaryOp),
            (pp.oneOf("< > <= >= == != ≠ ≤ ≥"), 2, pp.opAssoc.LEFT, BinaryComparison),
            ((BETWEEN | WITHIN, AND), 3, pp.opAssoc.LEFT, IntervalComparison),
            ((IN_RANGE_FROM, TO), 3, pp.opAssoc.LEFT, IntervalComparison),
            (NOT, 1, pp.opAssoc.RIGHT, UnaryNot),
            (AND | '∧', 2, pp.opAssoc.LEFT, BinaryComp),
            (OR | '∨', 2, pp.opAssoc.LEFT, BinaryComp),
            (('?', ':'), 3, pp.opAssoc.RIGHT, TernaryComp),
        ]
        ABS_VALUE_VERT = pp.Suppress("|")
        abs_value_expression = ABS_VALUE_VERT + arith_operand + ABS_VALUE_VERT
        def cvt_to_function_call(tokens):
            ret = pp.ParseResults(['abs']) + tokens
            ret['fn_name'] = 'abs'
            ret['args'] = tokens
            return [ret]
        abs_value_expression.addParseAction(cvt_to_function_call, function_node_class)

        arith_operand <<= pp.infixNotation((function_expression
                                            | abs_value_expression
                                            | string_operand
                                            | numeric_operand
                                            | bool_operand
                                            | var_name),
                                           self._added_operator_specs + base_operator_specs)
        rvalue = arith_operand.setName("rvalue")
        rvalue.setName("arithmetic expression")
        lvalue = var_name()

        value_assignment_statement = pp.delimitedList(lvalue)("lhs") + pp.oneOf("<- =") + pp.delimitedList(rvalue)("rhs")

        def eval_and_store_value(tokens):
            if len(tokens.lhs) > len(tokens.rhs):
                raise TypeError("not enough values given")
            if len(tokens.lhs) < len(tokens.rhs):
                raise TypeError("not enough variable names given")

            assignments = []
            for lhs_name, rhs_expr in zip(tokens.lhs, tokens.rhs):
                rval = LiteralNode([rhs_expr.evaluate()])
                var_name = lhs_name.name
                if (var_name not in identifier_node_class._assigned_vars
                        and len(identifier_node_class._assigned_vars) >= self.MAX_VARS):
                    raise Exception("too many variables defined")
                identifier_node_class._assigned_vars[var_name] = rval
                assignments.append(rval)
            return LiteralNode([assignments])

        value_assignment_statement.addParseAction(eval_and_store_value)
        value_assignment_statement.setName("assignment statement")

        formula_assignment_operator = "@="
        formula_assignment_statement = lvalue("lhs") + formula_assignment_operator + rvalue("rhs")
        formula_assignment_statement.setName("formula statement")

        def store_parsed_value(tokens):
            rval = tokens.rhs
            var_name = tokens.lhs.name
            if (var_name not in identifier_node_class._assigned_vars
                    and len(identifier_node_class._assigned_vars) >= self.MAX_VARS
                    or sum(sys.getsizeof(vv) for vv in identifier_node_class._assigned_vars.values())
                            > self.MAX_VAR_MEMORY):
                raise Exception("too many variables defined")
            identifier_node_class._assigned_vars[var_name] = rval
            return rval

        value_assignment_statement.addParseAction(RoundToEpsilon)
        lone_rvalue = rvalue().addParseAction(RoundToEpsilon)
        formula_assignment_statement.addParseAction(store_parsed_value)

        parser = (value_assignment_statement | formula_assignment_statement | lone_rvalue)

        # init _variable_map with any pre-defined values
        for varname, (varvalue, as_formula) in self._initial_variables.items():
            if as_formula:
                try:
                    parser.parseString("{} {} {}".format(varname, formula_assignment_operator, varvalue))
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


class BasicArithmeticParser(ArithmeticParser):
    def customize(self):
        def constrained_factorial(x):
            if not(0 <= x < 32768):
                raise ValueError("{!r} not in working 0-32,767 range".format(x))
            return math.factorial(int(x))

        super().customize()
        self.initialize_variable("pi", math.pi)
        self.initialize_variable("π", math.pi)
        self.initialize_variable("τ", math.pi * 2)
        self.initialize_variable("e", math.e)
        self.initialize_variable("φ", (1 + 5 ** 0.5) / 2)
        self.add_function('rnd', 0, random.random)
        self.add_function('randint', 2, random.randint)
        self.add_operator('°', 1, ArithmeticParser.LEFT, math.radians)
        self.add_operator("!", 1, ArithmeticParser.LEFT, constrained_factorial)
        self.add_operator("⁻¹", 1, ArithmeticParser.LEFT, lambda x: 1 / x)
        self.add_operator("²", 1, ArithmeticParser.LEFT, lambda x: x ** 2)
        self.add_operator("³", 1, ArithmeticParser.LEFT, lambda x: x ** 3)
        self.add_operator("√", 1, ArithmeticParser.RIGHT, lambda x: x ** 0.5)
        self.add_operator("√", 2, ArithmeticParser.LEFT, lambda x, y: x * y ** 0.5)
