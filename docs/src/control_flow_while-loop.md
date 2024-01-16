# While loop

A while loop is a control flow statement that allows code to be executed repeatedly based on a given boolean condition.

The condition is evaluated before each iteration of the loop. If the condition is true, the loop body is executed. If the condition is false, the loop body is skipped and the loop is terminated.

The condition is evaluated as a boolean expression. If the condition is not a boolean expression, it is converted to a boolean value. The following values are considered false:

The following example prints "Hello, World!" five times:

```orca
let n: i32 = 5;
while n > 0 {
    printf("Hello, World!\n");
    n -= 1;
}
```

The following example loops forever, because the condition is always true:

```orca
while true {
    printf("Hello, World!\n");
}
```
