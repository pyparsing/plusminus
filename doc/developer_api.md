## The Developer API


### A Simple Demo


### Parser definition methods

- `add_operator(operator_symbol, num_args, left_right_assoc, operator_function)`
- `add_variable(variable, value)`
- `add_function(function_name, num_args, function_method)`
  - if a function can accept any number of arguments, pass [`...`](https://docs.python.org/3/library/constants.html#Ellipsis) for the
    `num_args` argument (see the [`hypot`](https://docs.python.org/3/library/math.html#math.hypot) function)
  - if a function can accept different numbers of arguments, pass a tuple of all
    possible numbers of arguments a function can have for the `num_args` argument
    (see the [`log`](https://docs.python.org/3/library/math.html#math.log) function)

### Parser parse/evaluation methods

- `parse(expression)`
- `evaluate(expression)`
- `vars()`

### Parser attributes

- `customize()` method
- `LEFT` and `RIGHT`
- `MAX_VARS`
- `ident_letters`


### Usage Notes

Creating a custom ArithmeticParser class

Defining a new variable

Adding a new operator 

Adding a new function

Accessing evaluated results and variables from your code:

   ```python
   parser['xyz'] = 100
   parser.evaluate("√xyz") # Returns 10.0
   ```

Defining an operator that is both unary and binary


### SECURITY WARNINGS

  - Do not add functions that use or give access to [`eval`](https://docs.python.org/3/library/functions.html#eval), [`exec`](https://docs.python.org/3/library/functions.html#exec), [`compile`](https://docs.python.org/3/library/functions.html#compile), [`import`](https://docs.python.org/3/reference/simple_stmts.html#import), [`subprocess`](https://docs.python.org/3/library/subprocess.html#module-subprocess) or [`os`](https://docs.python.org/3/library/os.html#module-os)
  - If adding functions that start separate threads, limit the total number of threads that
    your parser will create at one time.
  - Be extremely careful if reading/writing to the file system
  - Take care when exposing access to an underlying database or server files
  - Some math functions may need to be constrained to avoid extended arithmetic processing (see 
    `constrained_factorial` in the [`BasicArithmeticParser`](https://github.com/pyparsing/plusminus/blob/master/doc/arithmetic_parser.md#the-core-basicarithmeticparser) as an example)
  - When populating web page elements with gathered input strings, be sure to escape potential quotation and control 
    characters (see [`bottle_repl.py`](https://github.com/pyparsing/plusminus/blob/master/plusminus/examples/bottle_repl.py) as an example)
  - Be aware that your functions may get called recursively, or in an
    endless loop
