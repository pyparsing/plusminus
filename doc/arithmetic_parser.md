## The core ArithmeticParser
- operators
- functions
- variables
- deferred evaluation


### Features:
- 6-function arithmetic (`+`, `-` , `*`, `/`, `//`, `**`)

- Unicode math operators (`×`, `÷`, `≠`, `≤`, `≥`, `∧`, `∨`, `∩`, `∪`, `∈`, `∉`)

- Additional operators

  - define sets using `{}` notation and `∩` (intersection) and
    `∪` (union) operators:
  
        {1, 2, 3}
        {}  # this is the empty set
        {1, 2, 3} ∪ {4, 5, 6}  # {1, 2, 3, 4, 5, 6}
        {1, 2, 3} ∩ {3, 4, 5}  # {3}
        {1, 2, 3} ∩ {4, 5, 6}  # {}

    use `∈` for "is element of" and `∉` for "is not element of"; `in` and `not in` 
    can also be used

  - `in range-specification`

    `in` operator takes a value and a range, specified with lower and upper values,
    enclosed in `()` characters (indicating exclusion of the boundary values) or `[]` 
    characters (indicating inclusion of the boundary values):
    
          x in (a, b)  -  a < x < b
          x in (a, b]  -  a < x <= b
          x in [a, b)  -  a <= x < b
          x in [a, b]  -  a <= x <= b
    
    This operator can be used on integers, reals, and strings.
        
  - `?:` ternary if-then-else
  - `not, and, or`
  - `mod` - modulo arithmetic - `8 mod 3 = 2`
  - `|x|` - absolute value - `abs(x)`

- Defined functions

  - `sgn`
  - `abs`
  - `round`
  - `trunc`
  - `ceil`
  - `floor`
  - `min`
  - `max`
  - `str`

- Multiple assignment

      a, b, c = 1, 2, a+b
      
  Results in:
  
      a = 1
      b = 2
      c = 3

- Deferred evaluation assignments

  Deferred evaluation assignments can be defined using `@=`:

      circle_area @= pi * r**2

  will re-evaluate `circle_area` using the current value of `r`.

  Calling functions in a deferred evaluation will run the function each time.
  But calling a deferred evaluation within another deferred evaluation will only
  evaluate the embedded deferred eval at assignment time.

- Variable names

  - Selected Unicode character ranges (Latin1, Greek)

        ABCDEFGHIJKLMNOPQRSTUVWXYZ
        abcdefghijklmnopqrstuvwxyz
        ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞß
        àáâãäåæçèéêëìíîïßðñòóôõöøùúûüýþÿªº
        ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
        αβγδεζηθικλμνξοπρστυφχψω

  - Trailing subscripts
  
        x₁, y₁ = 1, 2
        x₂, y₂ = 4, 0
        dist = hypot(x₂-x₁, y₂-y₁)


## The core BasicArithmeticParser

The BasicArithmeticParser class inherits all the features and behavior of the 
[`ArithmeticParser`](https://github.com/pyparsing/plusminus/blob/master/doc/arithmetic_parser.md#the-core-arithmeticparser) class. In addition, it also defines more operators and
functions, and common mathematical variables

- operators
  - `°` - degree (convert to radians)
  - `!` - factorial
  - `√` - square root (can be used a unary and binary operator)
  - `⁻¹` - superscript -1 - `x**(-1) or 1/x`
  - `²` - superscript 2 - `x**2`
  - `³` - superscript 3 - `x**3`
- functions
  - `sin`
  - `cos`
  - `tan`
  - `asin`
  - `acos`
  - `atan`
  - `sinh`
  - `cosh`
  - `tanh`
  - `rad`
  - `deg`
  - `ln`
  - `log`
  - `log2`
  - `log10`
  - `gcd`
  - `lcm`
  - `gamma`
  - `hypot`
  - `rnd`
  - `randint`
- variables
  - `pi` and `π` = 3.14159
  - `tau` and `τ` = 2π = 6.28319
  - `e` = 2.71828
  - `phi`, `φ` and `ϕ` = 1.61803

- Example expressions

    Expressions:

        2+3*11
        (2+3)*11
        sin(𝜋/2)
        sin(rad(30))
        sin(30°)
        |sin(-30°)|
        √5
        2√3

    Assignment expressions:

        m = 2
        x = 0
        b = 3
        y = m * x + b

    Formula assignments:

        y @= 2*x + 3
        x = 0
        y (evaluates to 3)
        x = 1
        y (evaluates to 5)

    Class implementation:

    ```python
    def log(x, y=10):
        if math.isclose(y, 2, abs_tol=1e-15):
            return math.log2(x)
        if math.isclose(y, 10, abs_tol=1e-15):
            return math.log10(x)
        return math.log(x, y)

    class BasicArithmeticParser(ArithmeticParser):
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
    ```
