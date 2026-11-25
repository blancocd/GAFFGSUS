<p align="center">
  <h1 align="center">Geometrically Accurate Feed‐Forward Gaussian Splatting for Unbounded Scenes</h1>
  <p align="center">
    <a href="https://blancocd.com/">César Díaz Blanco</a>
  </p>
  <p align="center">
    Supervisors: 
    <a href="https://niujinshuchong.github.io/">Zehao Yu</a>
    &middot;
    <a href="https://haofeixu.github.io/">Haofei Xu</a>
  </p>
  <p align="center">
    Examiners:
    <a href="https://www.cvlibs.net/">Andreas Geiger</a>
    &middot;
    <a href="https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html">Gerard Pons-Moll</a>
  </p>
  <h3 align="center">
    <a href="https://blancocd.com/ThesisWeb/MasterThesisCesarDiazBlanco_large.pdf">Thesis</a> | <a href="https://blancocd.com/ThesisWeb/">Project Page</a> | <a href="MODEL_ZOO.md">Model Zoo</a>
  </h3>
</p>

This thesis enables feed-forward Gaussian splatting models to achieve geometrically accurate 3D representations of unbounded scenes. Our method performs on par with costly per-scene optimization and surpasses state-of-the-art depth estimation models.

**Contributions:**
- **3D Smoothing Filter for feed-forward Gaussian splatting**: Current models ignore the spatial extent of predicted Gaussians, leading to unstable training. Applying a 3D filter to the predicted Gaussians prevents them from exceeding the Nyquist frequency and resolves numerical instabilities during training.
- **Edge-aware normal consistency loss**: Feed-forward splatting models allocate primitives equally across the scene decreasing the detail of normal consistency maps. We propose and edge-aware loss variant and succesfully steer the encoder to predict smoother surfaces despite the ambiguous normal renderings.
- **Fine-tuning of feed-forward models on new losses and decoders.**: Starting from an encoder trained with the gsplat decoder and photometric losses, we present a strategy to fine-tune it using the Gaussian Opacity Fields and Geometry-Grounded Gaussian Splatting decoders, alongside our edge-aware normal consistency loss.

The thesis also proposes a triangle encoder based on the DepthSplat architecture. Its repository is [here](https://github.com/blancocd/Master-Thesis-Triangle-Encoder).


## Installation

This codebase is developed with Python 3.12, PyTorch 2.8.0, and CUDA 12.6.

We recommend setting up a virtual environment (e.g., [conda](https://docs.anaconda.com/miniconda/) or [venv](https://docs.python.org/3/library/venv.html)) before installation:

```bash
# conda
conda create -y -n resplat python=3.12
conda activate resplat

# or venv
# python -m venv /path/to/venv/resplat
# source /path/to/venv/resplat/bin/activate

conda install nvidia/label/cuda-12.6.0::cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
pip install setuptools wheel ninja

# Install gsplat 1.5.3
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3

# Install pointops (kNN)
cd src/model/encoder/pointops && python setup.py install && cd ../../../..

# Install submodules
cd src/submodules

# Install GOF decoder
pip install ./gof-rasterization --no-build-isolation

# Install GGGS decoder
pip install ./gggs-rasterization --no-build-isolation

# Install Mesh Splatting decoder
pip install ./diff-triangle-mesh-rasterization --no-build-isolation

# Install GDel3D for fast Delaunay Triangulation
git clone https://github.com/blancocd/gDel3D-python
pip install ./gDel3D-python --no-build-isolation

# Install NVDiffRast
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

# Install TetraNeRF's (more stable) Triangulation
cd tetra-triangulation
conda install cmake
conda install conda-forge::gmp
conda install conda-forge::cgal
cmake .
make 
pip install -e .
cd ../../..

```

## Model Zoo

Pre-trained models are available in the [Model Zoo](MODEL_ZOO.md).

Download the weights and place (or symlink) them in the `pretrained` directory:

```bash
ln -s YOUR_MODEL_PATH pretrained
```

## Dataset Preparation

See [DATASETS.md](DATASETS.md) for detailed instructions on preparing RealEstate10K, DL3DV and ACID datasets.

Symlink the downloaded datasets to the `datasets` directory:

```bash
ln -s YOUR_DATASET_PATH datasets
```

## Training

The training scripts in [scripts/](scripts) contain the exact commands and hyperparameters used for all experiments. Please refer to them for detailed configurations.

## Evaluation

Evaluation scripts are also provided in [scripts/](scripts) for reproducing the results in our paper.

## Acknowledgements

Our codebase builds upon [ReSplat](https://github.com/cvg/resplat) as well as other several excellent open-source projects: [pixelSplat](https://github.com/dcharatan/pixelsplat), [MVSplat](https://github.com/donydchen/mvsplat), [MVSplat360](https://github.com/donydchen/mvsplat360), [UniMatch](https://github.com/autonomousvision/unimatch), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), [DepthSplat](https://github.com/cvg/depthsplat), [Pointcept](https://github.com/Pointcept/Pointcept), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [gsplat](https://github.com/nerfstudio-project/gsplat), [DL3DV](https://github.com/DL3DV-10K/Dataset), [Tanks and Temples](https://www.tanksandtemples.org/), [Mip-NeRF 360](https://jonbarron.info/mipnerf360/), [Gaussian Opacity Fields](https://github.com/autonomousvision/gaussian-opacity-fields), [Geometry-Grounded Gaussian Splatting](https://github.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting), [Mesh Splatting](https://github.com/meshsplatting/mesh-splatting), [gDel3D](https://github.com/ashwin/gDel3D), [nvdiffrast](https://github.com/NVlabs/nvdiffrast), and [TetraNeRF](https://github.com/jkulhanek/tetra-nerf). We thank all the authors for their great work.
