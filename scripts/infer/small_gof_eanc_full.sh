# SLURM_ARRAY_TASK_ID goes from 0 to 5 
# to run all configs

conda activate resplat
datasets=(mipnerf360 tnt dl3dv)
num_views=(8 16)

configs=(
    "mipnerf360 8"
    "mipnerf360 16"
    "tnt 8"
    "tnt 16"
    "dl3dv 8"
    "dl3dv 16"
)
config=${configs[$SLURM_ARRAY_TASK_ID]}
read -r dataset num_views <<< "$config"

echo "Job ID is $SLURM_JOB_ID"
echo "Running task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $dataset"

width=960
dataset_path=$dataset
case $dataset in
  "mipnerf360")
    height=640 ;;
  "tnt")
    height=540 ;;
  "dl3dv")
    width=480; height=270; dataset_path=dl3dv-480p-chunks ;;
esac

opencv_pose_format=false
case $dataset in
  "mipnerf360")
    case $num_views in
      "16")
        index_path=assets/mipnerf360_evaluation/mipnerf360_ctx_16v_tgt_15v_handcrafted.json ;;
      "8")
        index_path=assets/mipnerf360_evaluation/mipnerf360_ctx_8v_tgt_7v_handcrafted.json ;;
    esac ;;
  "tnt")
    case $num_views in
      "16")
        index_path=assets/tnt_evaluation/tnt_ctx_16v_tgt_15v_handcrafted.json ;;
      "8")
        index_path=assets/tnt_evaluation/tnt_ctx_8v_tgt_7v_handcrafted.json ;;
    esac ;;
  "dl3dv")
    opencv_pose_format=true
    case $num_views in
      "16")
        index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_120_ctx_16v_tgt_12v_subset.json ;;
      "8")
        index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v_subset.json ;;
    esac ;;
esac

output_dir=small_gof_eanc_full_${num_views}v

echo "Output directory is: $output_dir"
echo "Index path is: $index_path"
echo "Dataset is: $dataset"
echo "Image shape is: ${height},${width}"

export CPATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"

python -m src.main +experiment=dl3dv \
dataset.roots=[datasets/${dataset_path}] \
dataset.image_shape=[256,448] \
dataset.ori_image_shape=[${height},${width}] \
dataset.view_sampler.index_path=${index_path} \
mode=test \
dataset.test_chunk_interval=1 \
dataset/view_sampler=evaluation \
model.decoder.name=gggs \
model.encoder.render_with_quats=true \
model.encoder.nyquist_kernel_size=0.1 \
model.encoder.monodepth_vit_type=vits \
model.encoder.gaussian_regressor_channels=256 \
dataset.opencv_pose_format=${opencv_pose_format} \
checkpointing.pretrained_model=checkpoints/small_gof_eanc_full.ckpt \
output_dir=results/final/${output_dir} \
test.save_high_res_image=true \
test.save_high_res_mesh_image=true \
test.render_both_views=true \
test.save_rendered_depth=true \
test.save_depth_diff=true \
test.save_depth_normal=true \
test.save_gt_depth_normal=true \
test.save_depth_normal_diff=true \
test.save_mesh=true \
test.save_mesh_renders=true \
test.save_depth=true \
test.save_gaussian=true \
test.no_align_to_view=true \
test.save_gt_image=true \
test.save_image=true
