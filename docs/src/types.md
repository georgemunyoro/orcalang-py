# Types

The following is a list of the types available in Orca:

#### Integer types:

Integer types are defined as signed and unsigned integers of sizes 8, 16, 32 and 64 bits. The type specifiers are `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64` and `u64`. There is no generic `int` type.

#### Floating point types:

There are only 2 floating point types in Orca, `f32` and `f64`. There is no generic `float` type.

#### Boolean type:

The boolean type is defined as an unsigned integer of size 1 (`u1`). It can hold the values `0` and `1` only, defined as `false` and `true` respectively. The type specifier `bool` is just an alias for `u1`. Conditional expressions in Orca return a `bool`.

#### Character type:

The character type is defined as an unsigned integer of size 8 (`u8`). It can hold any value from `0` to `255`. The type specifier `char` is just an alias for `u8`.

#### Void type:

The void type is special in Orca, it is only usable as the return type of a function, the concept of values and variables of type `void` does not exist in Orca. You CAN **NOT** declare a variable to be of type `void`, or even a `void*`.

#### Function types:

Function types are declared in the form `(arg_types) -> return_type`. The parenthesis around the argument types are not optional, even if there are no arguments, or only one argument.

Examples:

- A function that takes no arguments and does not return a value:<br/>
  `() -> void`
- A function that takes 2 arguments of type `i32` and returns a `bool`:<br/>
  `(i32, i32) -> bool`

#### Pointer types:

Pointer types can be declared in the form `type*`.

Examples:

- `i32*`
- `bool**`
- `u16*`
- `() -> i32*`
- `(() -> i32)*`

  Note here the parenthesis around the function type. Wrapping a type in parenthesis does not change the type. There are no tuple types in Orca, do not confuse this with Rust's `()` type.

  This parenthesis notation exists to prevent ambiguity. Without parenthesis above, it would be:

  `() -> i32*`

  This is a very different type, a function that returns a pointer to an `i32` as opposed to a pointer to a function that returns an `i32`.

Note that the type specifier `void*` is **INVALID** in Orca, this is because the concept of values and variables of type `void` does not exist in Orca, therefore there is no need for a pointer to a `void` value.

If you find yourself needing a pointer to a `void` as your initial solution to a problem, you should probably rethink your design, but if you really need it, you can use a pointer to an integer, and cast it to it's original type when you need it.

#### Array types:

Array types can be declared in the form `type[length]` where `length` is a compile time constant integer expression. The length of an array must be greater than 0. The length is only needed when declaring an array, it is not part of the type. This is because arrays are passed by reference, so the length is not needed to access the array elements. The length is only needed to allocate the array on the stack.

Examples:

- `i32[10]` - An array of 10 `i32`s
- `bool[5]` - An array of 5 `bool`s
- `u16[100]` - An array of 100 `u16`s
- `() -> i32[10]` - A function that returns an array of 10 `i32`s
- `(() -> i32)[10]` - An array of 10 functions that return an `i32`. Note again here the parenthesis around the function type in order to prevent ambiguity.

#### Struct types:

Struct types can be declared in the form `struct { field1: type1; field2: type2; ... }`. The fields of a struct can be accessed using the `.` operator.

Examples:

- `struct { x: i32; y: i32; }` - A struct with 2 fields, `x` and `y`, both of type `i32`.
- `struct { x: i32; y: i32; }[10]` - An array of 10 structs with 2 fields, `x` and `y`, both of type `i32`. Note that there is no need for parenthesis around the struct type here, there is no room for ambiguity here in the grammar.

#### Enum types:

Enum types can be declared in the form `enum { variant1; variant2; ... }`. The variants of an enum can be accessed using the `.` operator.
