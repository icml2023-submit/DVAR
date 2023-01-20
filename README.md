# Is This Loss Informative? Speeding Up Textual Inversion with Deterministic Objective Evaluation
This repository contains the official code for our paper. 

## Setup

Before running the scripts, make sure to install the project dependencies:

```bash
pip install -r requirements.txt
```

In order to run experiments on Stable Diffusion, one needs to accept the model's license before downloading or using the weights. In our work, we use model version `v1-5`. To reproduce the presented results, you need to visit the [model page on Hugging Face Hub](https://huggingface.co/runwayml/stable-diffusion-v1-5), read the license and tick the checkbox if you agree. 

You have to be a registered user on the Hugging Face Hub, and you will also need to use an access token for the code to work. For more information on access tokens, please refer to [this section of the documentation](https://huggingface.co/docs/hub/security-tokens).

Run the following command to authorize:

```bash
huggingface-cli login
```

## Data
The full dataset with 92 concepts used in our work will be available later, after deanonymization of our work. For now our results can be reproduced on the dataset from original Textual Inversion paper, available [here](https://drive.google.com/drive/folders/1fmJMs25nxS_rSNqS5hTcRdLem_YQXbq5). Further we show an example of running our experiments with a "cat_statue" concept.

## Main experiments
The following commands can be used to reproduce the results presented in Table 1 of our paper. 
In order to reproduce results from Table 2, please run the "Baseline" method with varying `--init_strategy` parameter value (note that `--init_strategy=manual` requires specifying `--initialization_token`). 
To obtain numbers from Table 3, please also specify `--optimizer sam` in the baseline training script.
### Baseline

```bash
export DATA_DIR="path-to-cat_statue-images"
export CUDA_VISIBLE_DEVICES="your-gpus"

python textual_inversion.py \
  -m [sd/ldm] \
  --train_data_dir=$DATA_DIR \
  --placeholder_token="inversion-cat_statue"
  --init_strategy best \
  --train_batch_size=4 \
  --save_steps 500 \
  --max_train_steps 6100 \
  --learning_rate 5e-3 \
  --log_unscaled \
  --no_sample_before_start \
  --sample_frequency 500 \
  --n_val_prompts 8 \
  --variant vanilla \
  --mixed_precision fp16 \
  --logger wandb
```

### 700 iters

```bash
export DATA_DIR="path-to-cat_statue-images"
export CUDA_VISIBLE_DEVICES="your-gpus"

python textual_inversion.py \
  -m [sd/ldm] \
  --train_data_dir=$DATA_DIR \
  --placeholder_token="inversion-cat_statue"
  --init_strategy best
  --train_batch_size=4 \
  --save_steps 100 \
  --max_train_steps 700 \
  --learning_rate 5e-3 \
  --scale_lr \
  --no_sample_before_start \
  --sample_frequency 800 \
  --n_val_prompts 8 \
  --variant short_iters \
  --mixed_precision fp16 \
  --logger wandb
```

### CLIP-s

```bash
export DATA_DIR="path-to-cat_statue-images"
export CUDA_VISIBLE_DEVICES="your-gpus"

python textual_inversion.py \
  -m [sd/ldm] \
  --train_data_dir=$DATA_DIR \
  --placeholder_token="inversion-cat_statue"
  --init_strategy best
  --train_batch_size=4 \
  --save_steps 50 \
  --max_train_steps 6100 \
  --learning_rate 5e-3 \
  --scale_lr \
  --no_sample_before_start \
  --sample_frequency 50 \
  --n_val_prompts 8 \
  --variant clip_early_stopping \
  --early_stop_eps 0.05 \
  --early_stop_patience 5 \
  --mixed_precision fp16 \
  --logger wandb
```

### Ours
```bash
export DATA_DIR="path-to-cat_statue-images"
export CUDA_VISIBLE_DEVICES="your-gpus"

python textual_inversion.py \
  -m [sd/ldm] \
  --train_data_dir=$DATA_DIR \
  --placeholder_token="inversion-cat_statue"
  --init_strategy best
  --train_batch_size=4 \
  --eval_batch_size=4 \
  --save_steps 100 \
  --max_train_steps 6100 \
  --learning_rate 5e-3 \
  --scale_lr \
  --no_sample_before_start \
  --sample_frequency 6200 \
  --n_val_prompts 8 \
  --variant ours_early_stopping \
  --early_stop_eps 0.38 \
  --early_stop_patience 200 \
  --mixed_precision fp16 \
  --logger wandb
```
## Semi-deterministic experiments
This section shows how to set up experiments for Section 4.4 of our work.
The `exp_code` parameter is a binary code with each digit corresponding to each of the 5 main parameters of the evaluation batch (images/latents/noise/captions/timesteps) being deterministic (1) or random (0). 
For example, `exp_code=10101` means that every evaluation batch will have the same set of images, random noise tensors and timesteps for the diffusion process, but latents and captions will be resampled on each iteration.
To get the effective size of the validation batch as big as 512, one can vary the `--eval_gradient_accumulation_steps` parameter.

```bash
export DATA_DIR="path-to-dir-containing-images"
export CUDA_VISIBLE_DEVICES="your-gpus"

python eval_inversion.py \
  --max_train_steps 2500
  -m [sd/ldm] \
  --train_data_dir=$DATA_DIR \
  --placeholder_token="inversion-cat_statue"
  --init_strategy best
  --train_batch_size=4 \
  --eval_batch_size=8 \
  --save_steps 100 \
  --max_train_steps 6100 \
  --learning_rate 5e-3 \
  --scale_lr \
  --no_sample_before_start \
  --sample_frequency 50 \
  --exp_code 11110
  --n_val_prompts 8 \
  --mixed_precision fp16 \
  --eval_gradient_accumulation_steps 2 \
  --logger wandb
```
