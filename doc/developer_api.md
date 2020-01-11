## The Developer API

### Parser definition methods

- `add_operator(operator_symbol, num_args, left_right_assoc, operator_function)`
- `add_variable(variable, value)`
- `add_function(function_name, function_method, num_args)`

### Parser parse/evaluation methods

- `parse(expression)`
- `evaluate(expression)`
- `vars()`

### Usage Notes

Defining a new variable

Adding a new operator 

Adding a new function

Accessing evaluated results and variables from your code

- SECURITY WARNINGS
  - Do not add functions that give access to eval, exec, compile, import, subprocess, system, or Popen!
  - Be extremely careful if reading/writing to the file system
  - Take care when exposing access to an underlying database or server files
  - Some math functions may need to be constrained to avoid extended arithmetic processing (see constrained_factorial() for an example)
  - When populating web page elements with gathered input strings, be sure to escape potential quotation and control characters (see bottle_repl.py for an example)
  - Beware of functions or expressions that may get called recursively.

