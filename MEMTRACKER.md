| Commit                                   | Description                              | no_stages | with_missings | runtime (home) |
| ---------------------------------------- | ---------------------------------------- | --------- | ------------- | -------------- |
| b54a8d2679a97b5e3ce16f2244a3127269c9a37e | Baseline                                 | 2.4GiB    | 2.4GiB        | 27s            |
| afbabcd82c002765a5298d6fb763b22992182998 | No jitting at all                        | 3.9GiB    | 3.9GiB        | 36s            |
| 664504fb5518308d872e35b621caaa2358481589 | Jit \_scan_body                          | 3.9GiB    | 3.7GiB        | 37s            |
| 79d094bc40c2745067d44c953e935fcc526ce37d | jax.checkpoint on kalman_update          | 1.9GiB    | 1.8GiB        | 26s            |
| 00fcc5b99e0d19b7b743627b80a27224c458bc23 | jax.checkpoint on calculate_sigma_points | 1.9GiB    | 1.8GiB        | 25s            |
| 9b7796434e809ee34f60fdee6cf089d9ddf9baa6 | jax.checkpoint(prevent_cse=False)        | 1.9GiB    | 1.8GiB        | 25s            |
| a0ad3f0fefa7a08d7ab22647cdd031fb8eeb3821 | jax.checkpoint on kalman_predict         | 0.54GiB   | 0.49GiB       | 23s            |
