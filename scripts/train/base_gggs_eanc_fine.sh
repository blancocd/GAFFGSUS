conda activate resplat

# total number of steps is 800k
# finetuning requires half 
# 4 gpus 3 batches: 33333 steps
srun --export=ALL,PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
python -m src.main +experiment=dl3dv \
data_loader.train.batch_size=3 \
data_loader.train.num_workers=8 \
trainer.val_check_interval=0.25 \
trainer.eval_index=assets/tnt_evaluation/tnt_ctx_8v_tgt_7v_handcrafted.json \
dataset.roots="[datasets/dl3dv-480p-chunks]" \
dataset.view_sampler.num_context_views=8 \
dataset.view_sampler.num_target_views=6 \
dataset.view_sampler.min_distance_between_context_views=30 \
dataset.view_sampler.max_distance_between_context_views=60 \
dataset.view_sampler.initial_min_distance_between_context_views=24 \
dataset.view_sampler.initial_max_distance_between_context_views=40 \
dataset.image_shape=[256,448] \
trainer.max_steps=33334 \
optimizer.lr=1e-4 \
optimizer.lr_monodepth=1e-6 \
train.viz_geo_renders=true \
train.eval_depth=true \
dataset.load_depth=true \
dataset.load_normal=true \
model.decoder.name=gggs \
model.encoder.render_with_quats=true \
model.encoder.nyquist_kernel_size=0.1 \
model.encoder.init_use_checkpointing=true \
train.normal_cons_loss_weight=0.02 \
train.edge_aware_normal_cons=true \
wandb.tags=[dl3dv,base,256x448] \
output_dir=checkpoints/base_gggs_eanc_fine \
checkpointing.pretrained_model=checkpoints/shared/resplat-base-dl3dv-256x448-view8-1934a04c.pth \
checkpointing.no_strict_load=true \

# no_strict_load since the resplat base checkpoint includes refinement module
