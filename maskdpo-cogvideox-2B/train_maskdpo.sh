export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export MODEL_PATH="/inspire/qb-ilm/project/earthsimulation/public/qianwenxu/pretrained_models/CogVideoX-2b"
export CACHE_PATH="./cache"

export DATASET_PATH="/inspire/qb-ilm/project/earthsimulation/public/qianwenxu/maskdpo/maskdpo_train.json"

export SFTDATA_PATH="/inspire/qb-ilm/project/earthsimulation/public/qianwenxu/maskdpo/sft.json"

export VanillaDPODATA_PATH="/inspire/qb-ilm/project/earthsimulation/public/qianwenxu/dataset/video/cogvideo2b/cogvideo2B/cogvideo2b_dpo_label_5k.json"

export OUTPUT_PATH="/inspire/qb-ilm/project/earthsimulation/public/qianwenxu/maskdpo/maskdpo_lora/0308"




accelerate launch --config_file uncompiled_4.yaml \
  sammask_dpo.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir $CACHE_PATH \
  --enable_tiling \
  --enable_slicing \
  --instance_data_root $DATASET_PATH \
  --supervised_data_path $SFTDATA_PATH \
  --vanilladpo_data_path $VanillaDPODATA_PATH \
  --validation_prompt "The video begins with a close-up shot of a hand holding a black rectangular object against a plain white background. Beside it, a colorful cube constructed from small, round beads in various colors—red, green, yellow, blue, and gold—is placed on the surface. The hand holding the black object remains stationary for a moment before moving slightly to the left. As the hand moves, the black object is positioned closer to the colorful bead cube. The hand then lifts the black object, causing the colorful bead cube to tilt and eventually fall over onto its side. The video captures the dynamic interaction between the two objects, highlighting the contrast between the smooth, solid black rectangle and the vibrant, flexible bead cube. The scene concludes with the black object being held above the fallen bead cube, which lies on its side on the white surface." \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 30 \
  --beta_dpo 500 \
  --seed 42 \
  --rank 128 \
  --lora_alpha 256 \
  --mixed_precision fp16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --train_batch_size 4 \
  --num_train_epochs 6 \
  --checkpointing_steps 100 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to tensorboard