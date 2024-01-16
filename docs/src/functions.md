# Functions

In Orca, functions are defined using the `func` keyword. The syntax for defining a function is as follows:

```orca
func name(params) -> return_type {
    // code
}
```

A function can have zero or more parameters. Each parameter is a name followed by a type. The parameters are separated by commas.

The return type is the type of the value returned by the function. If the function does not return a value, the return type is `void`. The return type is **NOT** optional, even if the return type is `void`. Furthermore, a function of type `void` must not return a value.

The body of the function is a block of code that is executed when the function is called. The body of the function is **NOT** optional.

## Examples

- The following example defines a function named `add` that takes two parameters of type `i32` and returns a value of type `i32`:

  ```orca
  func add(x: i32, y: i32) -> i32 {
    return x + y;
  }
  ```

- The following example defines a function named `print` that takes one parameter of type `char*` and returns a value of type `void`:

  ```orca
  func print(s: char*) -> void {
    printf("%s\n", s);
  }
  ```
