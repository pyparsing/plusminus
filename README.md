# plusminus

The **plusminus** package provides a ready-to-run arithmetic parser and evaluator, 
based on [`pyparsing`](https://pyparsing-docs.readthedocs.io/en/latest/index.html)'s 
[`infix_notation`](https://pyparsing-docs.readthedocs.io/en/latest/pyparsing.html#pyparsing.infixNotation) 
helper method.

Strings containing 6-function arithmetic expressions can be parsed and evaluated using the 
[`ArithmeticParser`](https://github.com/pyparsing/plusminus/blob/master/doc/arithmetic_parser.md#the-core-basicarithmeticparser):

```python
from plusminus import BaseArithmeticParser

parser = BaseArithmeticParser()
print(parser.evaluate("2+3/10"))
```

The parser can also return an Abstract Syntax Tree of `ArithNode` objects:

```python
parsed_elements = parser.parse("2+3/10")
```

Arithmetic expressions are evaluated following standard rules for operator precedence, allowing for use of parentheses to override:

    ()
    |x|
    ∩ & ∪ | - ^ ∆ (set operations)
    **
    -
    * / // × ÷ mod
    + -
    < > <= >= == != ≠ ≤ ≥
    in ∈ ∉ (element in/not in set)
    not
    and ∧
    or ∨
    ? : (ternary)

Functions can be called:

    abs    ceil   max
    round  floor  str
    trunc  min    bool


The `BaseArithmeticParser` also supports assignment of variables:

    r = 5
    area = π × r²


This last expression could be assigned using  `@=` formula assignment:

    area @= π × r²


As `r` is updated, evaluating `area` will be reevaluated using the new value.


An `ArithmeticParser` class is also defined, with more extensive operators, 
including:

    !     - factorial  
    °     - degree-radian conversion
    √ ⁿ√  - square root and n'th root (2-9)
    ⁻¹  ⁰  ¹  ²  ³ - common exponents as superscripts

and additional pre-defined functions:

    sin    asin  rad    gcd
    cos    acos  deg    lcm
    tan    atan  ln     rnd
    sgn    sinh  log    randint
    gamma  cosh  log2
    hypot  tanh  log10

This parser class can be used in applications using algebra or trigonometry
expressions.

Custom expressions can be defined using a simple
[`API`](https://github.com/pyparsing/plusminus/blob/master/doc/developer_api.md).
Example parsers are included for other specialized applications
and domains:

- dice rolling (`"3d6 + d20"`)
- time delta expressions (`"today() + 2d + 12h"`)
- retail and business expressions (`"20% off of 19.99"`)
- combinatoric expressions (`"6C2"` or `"5P3"` )
 

These parsers can be incorporated into other
applications to support the safe evaluation of user-defined domain-specific
expressions.
