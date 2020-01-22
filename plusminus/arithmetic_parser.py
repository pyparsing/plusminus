#
# arithmetic_parsing.py
#
"""
arithmetic_parsing

arithmetic_parsing is a module that builds on the pyparsing infixNotation
helper method to build easy-to-code and easy-to-use parsers for parsing and
evaluating infix arithmetic expressions. arithmetic_parsing's ArithmeticParser
class includes separate parse and evaluate methods, handling operator
precedence, override with parentheses, presence or absence of whitespace,
built-in functions, and pre-defined and user-defined variables.
"""

from collections import namedtuple
import math
import operator
import random
import pyparsing as pp

ppc = pp.pyparsing_common
pp.ParserElement.enablePackrat()

__all__ = "ArithmeticParser BasicArithmeticParser expressions any_keyword".split()

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
    LEFT = pp.opAssoc.LEFT
    RIGHT = pp.opAssoc.RIGHT
    MAX_VARS = 1000

    class ArithmeticUnaryOp(UnaryNode):
        opns_map = {
            '+': lambda x: x,
            '-': operator.neg,
            '−': operator.neg,
        }

        def evaluate(self):
            return self.right_associative_evaluate(self.opns_map)

    class ArithmeticUnaryPostOp(UnaryNode):
        def evaluate(self):
            return self.left_associative_evaluate(self.opns_map)

    class ArithmeticBinaryOp(BinaryNode):
        opns_map = {
            '+': operator.add,
            '-': operator.sub,
            '−': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '**': math.pow,
            'mod': operator.mod,
            '×': operator.mul,
            '÷': operator.truediv,
        }

        def evaluate(self):
            if self.tokens[1] == '**':
                # parsed left-to-right, but evaluate right-to-left
                ret = self.tokens[-1].evaluate()
                for operand in self.tokens[-3::-2]:
                    op1 = operand.evaluate()
                    # rough guard against too large values in expression
                    if math.log10(abs(op1)) + math.log10(abs(ret)) > 8:
                        raise OverflowError("operands too large for expression")
                    ret = op1 ** ret
                return ret
            else:
                return self.left_associative_evaluate(self.opns_map)

    class IdentifierNode(ArithNode):
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
            opns_map = {
                '<': operator.lt,
                '<=': operator.le,
                '>': operator.gt,
                '>=': operator.ge,
                '==': operator.eq,
                '!=': operator.ne,
                '≠': operator.ne,
                '≤': operator.le,
                '≥': operator.ge,
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
            opns_map = {
                ('between', 'and'): (lambda a, b, c: b < a < c),
                ('within', 'and'): (lambda a, b, c: b <= a <= c),
                ('in_range_from', 'to'): (lambda a, b, c: b <= a < c),
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

        identifier_node_class = type('Identifier', (self.IdentifierNode,), {'_assigned_vars': self._variable_map})
        var_name.addParseAction(identifier_node_class)
        base_operator_specs = [
            ('-', 1, pp.opAssoc.RIGHT, self.ArithmeticUnaryOp),
            ('**', 2, pp.opAssoc.LEFT, self.ArithmeticBinaryOp),
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
                    and len(identifier_node_class._assigned_vars) >= self.MAX_VARS):
                raise Exception("too many variables defined")
            identifier_node_class._assigned_vars[var_name] = rval
            return rval

        formula_assignment_statement.addParseAction(store_parsed_value)

        parser = (value_assignment_statement | formula_assignment_statement | rvalue)

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
