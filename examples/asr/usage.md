
## Training

```
CUDA_VISIBLE_DEVICES=1 ./run.sh \
--stage 4 --stop_stage 4 \
--tag from-text-only-exhaustive-first-avgenc-train-all-bs5e05 \
--bs 16 --accum_grad 1 \
--exhaustion yes --use_audio yes \
--model exp/text-only-exhaustive/ --logging_steps 500 \
--fusion_place first --acoustic_encoder_type avg
```

## Perplexity
