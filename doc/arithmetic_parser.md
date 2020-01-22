## The core ArithmeticParser
- operators
- functions
- variables
- deferred evaluation


### Features:
- 5-function arithmetic (`+, - , *, /, **`)

- Unicode math operators (`×, ÷, ≠, ≤, ≥, ∧, ∨`)

- Additional operators

  - `between <lower> and <higher>`
  - `within <lower> and <higher>`
  - `in range from <lower> to <higher>`
  - `?:` ternary if-then-else
  - `not, and, or`
  - `mod`
  - `|x|` - absolute value - `abs(x)`

- Defined functions

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
  - `sgn`
  - `abs`
  - `round`
  - `trunc`
  - `ceil`
  - `floor`
  - `ln`
  - `log2`
  - `log10`
  - `gcd`
  - `lcm`
  - `gamma`
  - `hypot`

- Multiple assignment

      a, b, c = 1, 2, a+b

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

The `BasicArithmeticParser` class inherits all the features and behavior of the 
`ArithmeticParser` class. In addition, it also defines more operators and
functions, and common mathematical variables

- operators
  - `°` - degree (convert to radians)
  - `!` - factorial
  - `√` - square root (can be used a unary and binary operator)
  - `⁻¹` - superscript -1 - `x**(-1) or 1/x`
  - `²` - superscript 2 - `x**2`
  - `³` - superscript 3 - `x**3`
- functions
  - `rnd`
  - `randint`
- variables
  - `pi` and `π`
  - `τ`
  - `e`
  - `φ`

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
