
# PPL
```
CUDA_VISIBLE_DEVICES=6 ./run.sh \
--stage 6 --stop_stage 6 \
--use_audio yes --ppl yes \
--fusion_place first --acoustic_encoder_type conv \
--model_dir exp/from-text-only-exhaustive-first-convenc-train-ae-lr5e05/checkpoint-100000
```

# ASR decoding
```
CUDA_VISIBLE_DEVICES=2 ./run.sh \
--stage 6 --stop_stage 6 \
--use_audio yes --ppl no \
--fusion_place first --acoustic_encoder_type avg \
--model_dir exp/from-text-only-exhaustive-first-avgenc-train-all-lr5e-5/
```
