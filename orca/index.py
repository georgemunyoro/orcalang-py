from lark import Lark
from codegen import CodeGenerator
from compile import compile as orca_compile

with open("grammar.lark", "r") as grammar_file:
    with open("examples/main.orca", "r") as source_file:
        parser = Lark(grammar_file.read(), start="program")
        tree = parser.parse(source_file.read())

        print(tree.pretty())
        print("-" * 80)

        cg = CodeGenerator()
        codegen_res = cg.visit(tree)

        orca_compile(cg)
