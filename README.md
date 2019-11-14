# simple BP done in c
still work in progress.

```shell
$ gcc bp.c -o bp
$ ./bp 3d_data.csv > ddd.out
```

Please note that Eval is not finished and I sort of wrote an "parade" function that can only eval numbers, not vector of any demension.

```
// some part of the ddd.out (highlights, of course)
Round 0
loss: 0.499489
...
Round 999
loss: 0.000532
...
accuracy:488/500
```