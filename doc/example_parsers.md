## The example BusinessParser

- Operators
  - `%`
  - `of`

- Functions
  - `PV` - compute present value


    class BusinessParser(ArithmeticParser):
        def customize(self):
            super().customize()
            self.add_operator('%', 1, ArithmeticParser.LEFT, math.radians)
            self.add_operator("of", 2, ArithmeticParser.LEFT, lambda a, b: a * b)
            self.add_function('PV', random.randint, 2)


## The example Combinatorics Parser

- Operators
  - `P` permutations operator (`m P n` -> number of permutations of m items n at a time)
  - `C` combinations operator (`m C n` -> number of combinations of m items n at a time)


    class CombinatoricsParser(BasicArithmeticParser):
        def customize(self):
            super().customize()
            self.add_operator("P", 2, ArithmeticParser.LEFT, 
                                lambda a, b: a * b)
            self.add_operator("C", 2, ArithmeticParser.LEFT,
                                lambda a, b: a * b)
