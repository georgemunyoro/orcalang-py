# If statement

The if statement is used to make decisions. It is used to execute code only if a condition is true.

```orca
if condition {
    // Code to execute if condition is true
}
```

The condition must be a boolean expression. If the condition is true, the code inside the block is executed, otherwise it is skipped.

```orca
if 1 == 1 {
    // This code will be executed
}

if 1 == 2 {
    // This code will be skipped
}
```

You can also use an else block to execute code if the condition is false.

```orca
if 1 == 1 {
    // This code will be executed
} else {
    // This code will be skipped
}
```
