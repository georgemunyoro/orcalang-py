# Break statement

The `break` statement is used to terminate a loop early. It can only be used inside a loop body.

## Examples

The following example prints "Hello, World!" three times:

```orca
for (let i: i32 = 0; i < 5; i += 1) {
    if (i == 3) {
        break;
    }
    printf("Hello, World!\n");
}
```

The following example prints "Hello, World!" zero times, because the `break` statement is executed before the call to `printf`:

```orca
for (;;) {
    break;
    printf("Hello, World!\n");
}
```
