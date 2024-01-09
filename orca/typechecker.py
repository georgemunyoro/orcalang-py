from typing import List, Dict, Optional, TypeVar, Generic
from lark import Transformer, Tree

from scope import Scope


class ValueType:
    ...


class IntType(ValueType):
    def __str__(self):
        return "int"

    def __repr__(self):
        return str(self)


class GroupingType(ValueType):
    types: List[ValueType]

    def __init__(self, types: List[ValueType]):
        self.types = types

    def __str__(self):
        return f"{', '.join([str(i) for i in self.types])}"

    def __repr__(self):
        return str(self)


class VoidType(ValueType):
    def __str__(self):
        return "void"

    def __repr__(self):
        return str(self)


class FunctionType(ValueType):
    params: List[ValueType]
    return_type: ValueType

    def __init__(self, params: List[ValueType], return_type: ValueType):
        self.params = params
        self.return_type = return_type

    def __str__(self):
        return (
            f"({str(self.params)}) -> {str(self.return_type)}"
        )

    def __repr__(self):
        return str(self)


class TypeCheck(Transformer):
    scope: Scope[ValueType] = Scope[ValueType]()

    def program(self, values: Tree):
        for i in values:
            self.transform(i)
        return VoidType()

    def function_definition(self, tree):
        name, return_type, params, body = (None, None, None, None)
        if len(tree) == 4:
            name, return_type, params, body = tree
        else:
            name, return_type, body = tree

        self.scope.insert(
            name,
            FunctionType(
                params=GroupingType([]) if params is None else params,
                return_type=return_type,
            ),
        )
        print(self.scope.get(name))
        return VoidType()

    def type(self, values: List[Tree]):
        if values[0].data == "int":
            return IntType()
        return VoidType()
    
    def parameter_type_list(self, values):
        print(values)

    def parameter_type(self, values):
        ...

    def compound_statement(self, values):
        for statement in values:
            self.transform(statement)
        return VoidType()
