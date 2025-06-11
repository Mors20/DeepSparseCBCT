# 🚀 DeepSparseCBCT

This repository contains the official implementation of **DeepSparseCBCT**.

---


---

## 🛠️ Environment Setup (with Conda)

We recommend using [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for managing Python environments.
* Python >= 3.11.13
* PyTorch >= 2.2.0+cu12.1


### ✅ Create Conda Environment

```bash
git clone https://github.com/NVlabs/Sana.git
cd DeepSparseCBCT

conda env create -f environment.yml
# or you can install each components step by step following environment_setup.sh
```


## 🧪 Prepare Dataset


### 1️⃣ For Stage 1:
``` 
root_dir/ 
├── img001_highdose.nii.gz 
├── ... 

```
Update the path to `root_dir` in [`DeepSparseCBCT/config/DeepSparseCBCT_s1.yaml`](DeepSparseCBCT/config/DeepSparseCBCT_s1.yaml)

### 2️⃣ For Stage 2 and 3: 

``` 
input_dir/ 
├── img001_lowdose.nii.gz 
├── ... 

gt_dir/
├── img001_highdose.nii.gz 
├── ... 
```
Update the paths to input_dir and gt_dir in:
* DeepSparseCBCT/config/DeepSparseCBCT_s2.yaml
* DeepSparseCBCT/config/DeepSparseCBCT_s3.yaml


## 🧪 Training Pipeline
💡 Note: The following examples use torchrun for multi-GPU training. Adapt CUDA_VISIBLE_DEVICES, --nproc_per_node, and --master_port according to your setup.

### 1️⃣ Training Stage 1:
```bash
# Example (4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port=29513 \
  DeepSparseCBCT/train/train_DeepSparseCBCT_s1.py \
  --config DeepSparseCBCT/config/DeepSparseCBCT_s1.yaml
```

### 2️⃣ Training Stage 2:

```bash
# Example (4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port=29513 \
  DeepSparseCBCT/train/train_DeepSparseCBCT_s2.py \
  --config DeepSparseCBCT/config/DeepSparseCBCT_s2.yaml
```

### 3️⃣ Training Stage 3:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port 29513 # Example (4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port=29513 \
  DeepSparseCBCT/train/train_DeepSparseCBCT_s3.py \
  --config DeepSparseCBCT/config/DeepSparseCBCT_s3.yaml
```

## 🔍 Inference
```bash
python DeepSparseCBCT/inference/inference.py --input-dir your_input --output-dir your_output --ckpt-path Your_DeepSparseCBCT_s3.ckpt
```

