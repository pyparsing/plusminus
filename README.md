# plusminus

The **plusminus** package provides a ready-to-run arithmetic parser and evaluator, based on [`pyparsing`](https://github.com/pyparsing/pyparsing)'s 
[`infixNotation`](https://github.com/pyparsing/pyparsing/blob/master/pyparsing/helpers.py#L608) helper method.

Strings containing 5-function arithmetic expressions can be parsed and evaluated using the [`BasicArithmeticParser`](https://github.com/pyparsing/plusminus/blob/master/plusminus/plusminus.py#L1105):

```python
from plusminus import BasicArithmeticParser

parser = BasicArithmeticParser()
print(parser.evaluate("2+3/10"))
```

The parser can also return an Abstract Syntax Tree of `ArithNode` objects:

```python
parsed_elements = parser.parse("2+3/10")
```

Arithmetic expressions are evaluated following standard rules for operator precedence, allowing for use of parentheses `()`'s 
to override:

    ()
    ∩ (set intersection)
    ∪ (set union)
    -
    **
    * / × ÷ mod
    + -
    < > <= >= == != ≠ ≤ ≥
    in ∈ ∉
    not
    and ∧
    or ∨
    ? : (ternary)

Functions can be called:

      sgn    min  asin  rad    gcd
      abs    max  acos  deg    lcm
      round  str  atan  ln     gamma
      trunc  sin  sinh  log    hypot
      ceil   cos  cosh  log2   nhypot
      floor  tan  tanh  log10  rnd


The Basic ArithmeticParser also supports assignment of variables:

    r = 5
    area = π × r²


This last expression could be assigned using '@=' formula assignment:

    area @= π × r²


As `r` is updated, evaluating `area` will be reevaluated using the new value.


Custom expressions can be defined using a simple [`API`](https://github.com/pyparsing/plusminus/blob/master/doc/developer_api.md). Example parsers
are included for dice rolling, combination/permutation expressions, and 
common business calculations. These parsers can be incorporated into
other applications to support the safe evaluation of user-defined 
domain-specific expressions.
