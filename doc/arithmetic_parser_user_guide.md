# User Guide - Working with the Basic Arithmetic Parser

## Introduction

## Simple Expressions

### Operators

      √   **   +   ==  ∉    ∧
      ³   *    -   !=  ∩    or
      ²   /    <   ≠   ∪    ∨
      ⁻¹  mod  >   ≤   in   ?:
      !   ×    <=  ≥   not  |absolute-value|
      °   ÷    >=  ∈   and

- 5-function arithmetic (`+, - , *, /, **`)

- Unicode math operators (`×, ÷, ≠, ≤, ≥, ∧ (`and`), ∨ (`or`)`, `∩`, `∪`, `∈`, `∉`)

- Additional operators

  - `in range-expression`
    - where `range-expression` using enclosing `(`, `)`, `[`, and `]`
      symbols:
      
          b in (a, c) evaluates a < b < c
          b in [a, c) evaluates a <= b < c
          b in (a, c] evaluates a < b <= c
          b in [a, c] evaluates a <= b <= c

    - can be negated using `not in`

  - `?:` ternary if-then-else (`condition ? true-value : false-value`)
    - conditions that evaluate to 0 or False will be treated as False
    - conditions that evaluate to non-zero or True will be treated as True

  - set operators `∩`, `∪`, `∈`, `∉`
    - sets are defined using `{`, `}` symbols    
    - `∩` is intersection, `∪` is union
    - `∈` is `is element of`, `∉` is `is not element of`

          1 ∈ {1, 2, 3} ∪ {10, 11, 12} evaluates True
          1 ∉ {1, 2, 3} ∩ {10, 11, 12} evaluates True

    - `{}` is the empty set
    - `in` and `not in` are equivalent to `∈` and `∉`

  - `not, and, or`
  - `mod`
  - `|x|` - absolute value - `abs(x)`
  - `°` - degree (convert to radians)
  - `!` - factorial
  - `√` - square root (can be used as a unary or binary operator)
  - `⁻¹` - superscript (-1) - `x**(-1) or 1/x`
  - `²` - superscript 2 - `x**2`
  - `³` - superscript 3 - `x**3`

#### Precedence of Operations

    |x|
    {}
    ∩
    ∪
    °
    !
    ³ ² ⁻¹
    √
    !    
    leading '-'
    **
    * / × ÷ mod
    + -
    < > <= >= == != ≠ ≤ ≥
    in, not in, ∈ ∉
    not
    and ∧
    or ∨
    ?:
    |x|

### Functions

      sgn    min  asin  rad    lcm
      abs    max  acos  deg    gamma
      round  str  atan  ln     hypot
      trunc  sin  sinh  log2   nhypot
      ceil   cos  cosh  log10  rnd
      floor  tan  tanh  gcd    

### Variables

     'e': 2.718281828459045
     'pi': 3.141592653589793
     'π': 3.141592653589793
     'τ': 6.283185307179586
     'φ': 1.618033988749895

### Assignment statements

You can define your own variables using assignment statements:

      x = 10000
      y = √x
      θ = 45°

Variable names may be defined using any of the following characters:

        A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
        a b c d e f g h i j k l m n o p q r s t u v w x y z _
        À Á Â Ã Ä Å Æ Ç È É Ê Ë Ì Í Î Ï Ð Ñ Ò Ó Ô Õ Ö Ø Ù Ú Û Ü Ý Þ ß
        à á â ã ä å æ ç è é ê ë ì í î ï ß ð ñ ò ó ô õ ö ø ù ú û ü ý þ ÿ ª º
        Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ Σ Τ Υ Φ Χ Ψ Ω
        α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω
        0 1 2 3 4 5 6 7 8 9

(Numeric digits may not be used as the first character of a name.)

Trailing subscripts using subscript digits `₀₁₂₃₄₅₆₇₈₉` can be added to any variable 
name (the subscripts do
not reflect any ordering or array storage of values, they are merely additional 
characters you may use at the end of a variable name):
  
        x₁ = 1
        y₁ = 2


### Multi-value assignment

Multiple assignments can be made using lists of variable names and
corresponding lists of expressions (lists must be of matching lengths).

    x₁, y₁ = 1, 2
    x₂, y₂ = 4, 0
    a, b, c = 1, 2, a+b

Expressions and assignments are made respectively from left-to-right. The
third expression above is evaluated in this order:

    a = 1
    b = 2
    c = a + b


### Formula assignments

A formula assignment uses `@=` to assign an expression to a variable:

        area @= π × r²

As the value of `r` is updated, each evaluation of `area` will recalculate the given formula.

Note: Formula assignments may only reference variables and functions, but
not other formula variables.


### Example: Trigonometric functions (converting from degrees to radians using the `°` operator)

    sin(pi/2)
    sin(π/2)
    sin(-π/2)
    sin(30)
    sin(rad(30))
    sin(30°)

### Example: Converting from Celsius to Fahrenheit

    c_temp = 37
    F @= (c_temp × 9/5) + 32

### Example: Distance formula

    x₁, y₁ = 1, 2
    x₂, y₂ = 4, 0
    dist @= √((x₂-x₁)² + (y₂-y₁)²)

### Example: Solving for roots of a quadratic polynomial (_`ax² + bx + c`_)

    a,b,c = 1,1,-12
    r₀ @= (-b + √(b² - 4×a×c)) / (2×a)
    r₁ @= (-b - √(b² - 4×a×c)) / (2×a)
