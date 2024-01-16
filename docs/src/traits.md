# Traits

Orca is not an object-oriented language. However, Orca does support traits. Traits are similar to interfaces in other languages. Traits are defined using the `trait` keyword. The syntax for defining a trait is as follows:

```orca
trait name {
    // code
}
```

The body of the trait is a block of code that defines the methods of the trait. The body of the trait is **NOT** optional.

The body of the trait should contain one or more method definitions. Each method definition is a function definition without a body. The return type of the method is the type of the value returned by the method. If the method does not return a value, the return type is `void`. The return type is **NOT** optional, even if the return type is `void`. Furthermore, a method of type `void` must not return a value.

The following example defines a trait named `Printable` that defines a method named `print` that takes no parameters and returns a value of type `void`:

```orca
trait Printable {
  func print() -> void;
}
```

When defining a trait, you can use the `Self` type to refer to the type that implements the trait. The `Self` type is similar to the `this` keyword in other languages. The `Self` type is only available inside the body of the trait, and it can only be used in method definitions. It also should only be the first parameter of a method definition. When no `self: Self*` parameter is specified, method is assumed to be a static method.

```orca
  trait Addable {
      func add(self: Self*, x: i32) -> i32;
  }
```

You can also use the `Self` type to refer to the type that implements the trait when defining the return type of a method. The following example defines a trait named `Addable` that defines a method named `add` that takes one parameter of type `i32` and returns a value of type `Self`:

```orca
  trait Addable {
      func add(self: Self*, x: i32) -> Self;
  }
```

## Implementing traits

Traits can be implemented for structs. To implement a trait for a struct, use the `impl` keyword. The syntax for implementing a trait is as follows:

```orca
impl name for type {
    // code
}
```

The `name` is the name of the trait. The `type` is the type that implements the trait. The body of the implementation is a block of code that defines the methods of the trait. The body of the implementation is **NOT** optional. **ALL** methods defined in the trait must be implemented.

The following example defines a struct named `Point` that implements the `Printable` trait:

```orca
  struct Point {
      x: i32,
      y: i32,
  }

  impl Printable for Point {
      func print(self) -> void {
          printf("(%d, %d)\n", self.x, self.y);
      }
  }
```

## Examples

The following example defines a struct named `Point` that implements the `Addable` trait:

```orca
  struct Point {
      x: i32,
      y: i32,
  }

  impl Addable for Point {
      func add(self, x: i32) -> i32 {
          return Point { x: self.x + x, y: self.y + x };
      }
  }
```
