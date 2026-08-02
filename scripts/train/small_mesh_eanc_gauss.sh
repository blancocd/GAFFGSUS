conda activate resplat

# total number of steps is 400k
# finetuning for mesh testing for a quarter 
# 4 gpus 1 batch: 25k steps
srun --export=ALL,PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
python -m src.main +experiment=dl3dv \
data_loader.train.batch_size=1 \
trainer.val_check_interval=0.25 \
trainer.eval_index=assets/tnt_evaluation/tnt_ctx_8v_tgt_7v_handcrafted.json \
dataset.roots="[datasets/dl3dv-480p-chunks]" \
dataset.view_sampler.num_context_views=8 \
dataset.view_sampler.num_target_views=6 \
dataset.view_sampler.min_distance_between_context_views=24 \
dataset.view_sampler.max_distance_between_context_views=45 \
dataset.view_sampler.initial_min_distance_between_context_views=20 \
dataset.view_sampler.initial_max_distance_between_context_views=30 \
dataset.image_shape=[256,448] \
trainer.max_steps=25000 \
model.encoder.monodepth_vit_type=vits \
model.encoder.gaussian_regressor_channels=256 \
optimizer.lr=1e-4 \
optimizer.lr_monodepth=1e-6 \
train.eval_depth=true \
dataset.load_depth=true \
dataset.load_normal=true \
model.decoder.name=gggs \
model.encoder.render_with_quats=true \
model.encoder.nyquist_kernel_size=0.1 \
model.encoder.init_use_checkpointing=true \
wandb.tags=[dl3dv,base,256x448] \
output_dir=checkpoints/small_mesh_eanc_gauss \
checkpointing.pretrained_model=checkpoints/small_gggs_eanc_fine.ckpt \
checkpointing.no_strict_load=true \
train.mesh_nvs_weight=3. \
train.triangle_sigma=0. \
train.opacity_threshold=0.1 \
train.remove_duplicates=false \
train.vertex_corner_type=octahedron \
train.viz_geo_renders_gaussians=true \
train.viz_geo_renders=true \
train.normal_cons_loss_weight=0.02 \
train.edge_aware_normal_cons=true \
