# plusminus

The plusminus package provides a ready-to-run arithmetic parser and evaluator, base on pyparsing's 
`infixNotation` helper method.

Strings containing 5-function arithmetic expressions can be parsed and evaluated using the BasicArithmeticParser:

    from plusminus import BasicArithmeticParser
    
    parser = BasicArithmeticParser()
    print(parser.evaluate("2+3/10"))

The parser can also return an Abstract Syntax Tree of `ArithNode` objects:

    parsed_elements = parser.parse("2+3/10")

Arithmetic expressions are evaluated following standard rules for operator precedence, allowing for use of parentheses `()`'s 
to override:

    ()
    -
    **
    * / mod
    + -
    < > <= >= == !=
    between-and within-and "in range from"-to (ternary) (between is exclusive, within is inclusive, and `in range from` is
      inclusive lower-bound and exclusive upper-bound)
    not
    and
    or
    ? : (ternary)

The Basic ArithmeticParser also supports assignment of variables:

    m = 10
    a = 9.8
    F = m * a

This last expression could be assigned using '@=' formula assignment:

    F @= m * a

As `m` and `a` are updated, evaluting `F` will reevaluate using the new values.

