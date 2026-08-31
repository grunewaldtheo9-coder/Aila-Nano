# Aila Nano data-scaling report

- Model parameters: **51,393,024**
- Dataset version: `aila_pretrain_v1_tinystories`
- Data points: 3 (measured, not interpolated)

## Results

| Tokens seen | Epochs | Tokens/param | Best val loss | Val PPL | Final val loss | Time (s) | Tokens/sec |
|---|---|---|---|---|---|---|---|
| 503,808 | 0.0289 | 0.0098 | 4.4375 | 84.5649 | 4.4411 | 903.2 | 557.8 |
| 1,007,616 | 0.0579 | 0.0196 | 3.7719 | 43.4636 | 3.8441 | 1424.4 | 707.4 |
| 2,002,944 | 0.115 | 0.039 | 3.2552 | 25.9241 | 3.346 | 2806.6 | 713.6 |

## Validation loss vs training tokens

```
  y: best_val_loss [3.2552 .. 4.4375]   x: tokens_seen [503,808 .. 2,002,944]
|*                                                 
|                                                  
|                                                  
|                                                  
|                                                  
|                                                  
|                *                                 
|                                                  
|                                                  
|                                                  
|                                                  
|                                                 *
+--------------------------------------------------
```

## Validation loss vs tokens per parameter

```
  y: best_val_loss [3.2552 .. 4.4375]   x: tokens_per_parameter [0.0098 .. 0.039]
|*                                                 
|                                                  
|                                                  
|                                                  
|                                                  
|                                                  
|                *                                 
|                                                  
|                                                  
|                                                  
|                                                  
|                                                 *
+--------------------------------------------------
```
