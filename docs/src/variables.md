# Variables

Variables in Orca are defined using the `let` keyword. The syntax for defining a variable is as follows:

```orca
let name: type = value;
```

In orca, all values are immutable by default. To make a variable mutable, use the `mut` keyword:

```orca
let mut name: type = value;
```

The `name` is the name of the variable. The `type` is the type of the variable. The `value` is the initial value of the variable.

## Examples

- The following example defines a variable named `x` of type `i32` with the value `42`:

  ```orca
  let x: i32 = 42;
  ```

- The following example defines a variable named `y` of type `f64` with the value `3.14`:

  ```orca
  let y: f64 = 3.14;
  ```

- The following example defines a variable named `z` of type `bool` with the value `true`:

  ```orca
  let z: bool = true;
  ```

- The following example defines a variable named `s` of type `char*` with the value `"Hello, World!"`:

  ```orca
  let s: char* = "Hello, World!";
  ```

- The following example defines a variable named `arr` of type `i32[5]` with the value `[1, 2, 3, 4, 5]`:

  ```orca
  let arr: i32[5] = [1, 2, 3, 4, 5];
  ```

- The following example defines a variable named `obj` of type `struct { x: i32, y: i32 }` with the value `{x: 1, y: 2}`:

  ```orca
  let obj: struct { x: i32, y: i32 } = {x: 1, y: 2};
  ```

- The following example defines a variable named `fn` of type `(i32, i32) -> i32` with the value of a function named `add`:

  ```orca
  let fn: (i32, i32) -> i32 = add;
  ```

- The following example defines a variable named `ptr` of type `i32*` with the value of a pointer to the variable `x`:

  ```orca
  let ptr: i32* = &x;
  ```
