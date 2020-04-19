## The Developer API


### A Simple Demo


### Parser definition methods

- `add_operator(operator_symbol, num_args, left_right_assoc, operator_function)`
- `add_variable(variable, value)`
- `add_function(function_name, num_args, function_method)`
  - if a function can accept a variable number of arguments, pass `...` for the
    `num_args` argument; see the `nhypot` function in the `BasicArithmeticParser`

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

Accessing evaluated results and variables from your code

Defining an operator that is both unary and binary


### SECURITY WARNINGS

  - Do not add functions that use or give access to `eval`, `exec`, `compile`, `import`, `subprocess`, 
    `system`, or `Popen`!
  - If adding functions that start separate threads, limit the total number of threads that
    your parser will create at one time.
  - Be extremely careful if reading/writing to the file system
  - Take care when exposing access to an underlying database or server files
  - Some math functions may need to be constrained to avoid extended arithmetic processing (see 
    `constrained_factorial()` in the `BasicArithmeticParser` for an example)
  - When populating web page elements with gathered input strings, be sure to escape potential quotation and control 
    characters (see `bottle_repl.py` for an example)
  - Be aware that your functions may get called recursively, or in an
    endless loop
