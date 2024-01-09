from typing import Generic, Dict, Optional, TypeVar

T = TypeVar("T")


class Scope(Generic[T]):
    data: Dict[str, T] = dict({})
    parent: Optional["Scope"] = None

    def __init__(self, parent: Optional["Scope"] = None):
        self.parent = parent
        self.data = dict({})

    def get(self, key: str) -> Optional[T]:
        if key not in self.data:
            if self.parent is not None:
                return self.parent.get(key)
            return None
        return self.data[key]

    def insert(self, key: str, val: T):
        self.data[key] = val

    def print(self, indent: int = 0):
        indent_str = " " * indent
        if self.data:
            print(f"{indent_str}Scope Data:")
            for key, value in self.data.items():
                print(f"{indent_str}  {key}: {value}")
        else:
            print(f"{indent_str}Scope (empty)")

        if self.parent:
            print(f"{indent_str}Parent Scope:")
            self.parent.print(indent + 2)
