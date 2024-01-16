# For loop

A for loop is a control flow statement that allows code to be executed repeatedly based on a given boolean condition.

The syntax of a for loop is as follows:

```orca
for (let init; cond; inc) {
    // code
}
```

The `init` statement is executed once before the loop starts. It is typically used to initialize variables.

The `cond` statement is evaluated before each iteration of the loop. If the condition is true, the loop body is executed. If the condition is false, the loop body is skipped and the loop is terminated.

The `inc` statement is executed after each iteration of the loop. It is typically used to increment variables.

## Examples

The following example prints "Hello, World!" five times:

```orca
for (let i: i32 = 0; i < 5; i += 1) {
    printf("Hello, World!\n");
}
```

The following example loops forever, because the condition is always true:

```orca
for (;;) {
    printf("Hello, World!\n");
}
```

For loops can be used to iterate over arrays:

```orca
let arr: i32[5] = [1, 2, 3, 4, 5];
for (let i: i32 = 0; i < 5; i += 1) {
    printf("%d\n", arr[i]);
}
```
