# Continue statement

The `continue` statement is used to skip the rest of the loop body and continue with the next iteration of the loop. It can only be used inside a loop body.

## Examples

The following example prints "Hello, World!" four times:

```orca
for (let i: i32 = 0; i < 5; i += 1) {
    if (i == 3) {
        continue;
    }
    printf("Hello, World!\n");
}
```

The following example loops forever, but prints "Hello, World!" zero times, because the `continue` statement is executed before the call to `printf`:

```orca
for (;;) {
    continue;
    printf("Hello, World!\n");
}
```
