## Closing the Train-Test Gap in World Models for Gradient-Based Planning
> Arjun Parthasarathy\*, Nimit Kalra\*, Rohun Agrawal\*,  
> Yann LeCun, Oumayma Bounou, Pavel Izmailov, Micah Goldblum

<div align="center">

<img alt="Our Method" width="75%" src="https://github.com/user-attachments/assets/abd293be-cb05-4210-b559-a3f5682d7cd8"/>

<p align="center">
  <a href="https://arxiv.org/abs/2512.09929" target="_blank">📄 Paper</a> •
  <a href="#checkpoints">🤖 Models</a> •
  <a href="#datasets">🗂️ Data</a>
</p>
</div>

### Checkpoints
Pretrained world model checkpoints are provided by [DINO-WM](https://github.com/gaoyuezhou/dino_wm) and can be downloaded [here](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28) under `checkpoints`.

<div align="center">
<img alt="Our Method" width="75%" src="https://github.com/user-attachments/assets/2ad42535-1e5b-474a-9217-efba54f24b18"/>
</div>

**Online/Adversarial World Modeling Checkpoints.** Below, we provide checkpoints obtained after applying our methods to the pretrained DINO-WM checkpoints. These correspond to the results in Table 1. We recommend using [`gdown`](https://github.com/wkentaro/gdown) to download these to your machine.
|         Method    |                                                        PushT                                                       |                                                        PointMaze                                                       |                                                        Wall                                                       |
|-------------:|:------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|
|      Online | [`pusht.online.6000`](https://drive.google.com/file/d/14rn3dD4ezLC4Qh5Xfwrlwe7_Wlxohecu/view?usp=sharing)      | [`pointmaze.online.100`](https://drive.google.com/file/d/1bpX5TaW4DhgfKDj7iwQ5r2fk4teVCUiW/view?usp=sharing)       | [`wall.online.full`](https://drive.google.com/file/d/12BXoo7WYJbhy976Ee_e43NiQfHBk82tT/view?usp=sharing)      |
| Adversarial | [`pusht.adversarial.full`](https://drive.google.com/file/d/1le0G8zJYRJTz-2lyOFtn6QhCqE6cuZwo/view?usp=sharing) | [`pointmaze.adversarial.full`](https://drive.google.com/file/d/1jB24wVsw7dRy9PwpV0DhzcWQNmXMLtsl/view?usp=sharing) | [`wall.adversarial.full`](https://drive.google.com/file/d/1fWdI-dA1HOsPGg-CdQZPrXkSw8LoDELO/view?usp=sharing) |

### Installation

Our code is adapted from [DINO-WM](https://github.com/gaoyuezhou/dino_wm). Please refer to their repo to any additional installation and setup instructions. See [here](https://gist.github.com/qw3rtman/e50d5414aa5c6435dad87eec7b7a7c6f) for Modal specific instructions.

First clone the repo and create a Python environment for dependencies.
```bash
git clone https://github.com/qw3rtman/robust-world-model-planning.git
cd robust-world-model-planning
conda env create -f environment.yaml
conda activate robust_wm
```

Then, install Mujoco.
```bash
mkdir -p ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/.mujoco/
cd ~/.mujoco
tar -xzvf mujoco210-linux-x86_64.tar.gz
```

```bash
# Mujoco Path. Replace `<username>` with your actual username if necessary.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<username>/.mujoco/mujoco210/bin

# NVIDIA Library Path (if using NVIDIA GPUs)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

Notes:
- For GPU-accelerated simulations, ensure the NVIDIA drivers are correctly installed.
- If you encounter issues, confirm that the paths in your `LD_LIBRARY_PATH` are correct.
- If problems persist, refer to these GitHub issue pages for potential solutions: [openai/mujoco-py#773](https://github.com/openai/mujoco-py/issues/773), [ethz-asl/reinmav-gym#35](https://github.com/ethz-asl/reinmav-gym/issues/35).


### Datasets
We use training data collected by Zhou et al. in DINO-WM [here](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28). Once the datasets are downloaded, unzip them.

Set an environment variable pointing to your dataset folder:
```bash
# Replace /path/to/data with the actual path to your dataset folder.
export DATASET_DIR=/path/to/data
```
Setup the dataset folder with the following structure:
```
data
├── point_maze
├── pusht_noise
└── wall_single
```

### Training Robust World Models
To finetune a base world model with either Online World Modeling or Adversarial World Modeling, run `train.py` with the appropriate overrides for the environment and method:

```bash
# PushT
python train.py --config-name train.yaml env=pusht ckpt_path=./outputs/pusht/ method=online
python train.py --config-name train.yaml env=pusht ckpt_path=./outputs/pusht/ method=adversarial

# PointMaze
python train.py --config-name train.yaml env=point_maze ckpt_path=./outputs/point_maze/ method=online
python train.py --config-name train.yaml env=point_maze ckpt_path=./outputs/point_maze/ method=adversarial

# Wall
python train.py --config-name train.yaml env=wall ckpt_path=./outputs/wall_single/ num_hist=1 method=online
python train.py --config-name train.yaml env=wall ckpt_path=./outputs/wall_single/ num_hist=1 method=adversarial
```
Note that the base checkpoint used will be `<ckpt_path>/checkpoints/model_latest.pth`.
During finetuning, checkpoints will be saved to `<ckpt_path>/<method>/<date>/<time>/checkpoints`.

### Planning with Robust World Models
To plan with a finetuned world model, run `plan.py` with the appropriate config file for the environment. Set `ckpt_path` to the path that points to the parent directory of the `checkpoints` directory containing the checkpoint you want to use. 

```bash
# PushT
python plan.py --config-name plan_pusht.yaml ckpt_path=./outputs/pusht/<method>/<date>/<time>/

# PointMaze
python plan.py --config-name plan_point_maze.yaml ckpt_path=./outputs/point_maze/<method>/<date>/<time>/

# Wall
python plan.py --config-name plan_wall.yaml ckpt_path=./outputs/wall_single/<method>/<date>/<time>/
```
Planning results and visualizations will be saved to `plan_outputs/<env>/<current_time>`.

For using our paper's checkpoints, replace ckpt_path with `ckpt_path=./outputs/<env>/<method>`.

## Citation
```
@article{parthasarathy2025closing,
    title   = {Closing the Train–Test Gap in World Models for Gradient-Based Planning},
    author  = {Arjun Parthasarathy, Nimit Kalra, Rohun Agrawal, Yann LeCun, Oumayma Bounou, Pavel Izmailov, Micah Goldblum},
    journal = {arXiv preprint arXiv:2512.09929},
    url     = {https://arxiv.org/abs/2512.09929},
    year    = {2025}
}
```
