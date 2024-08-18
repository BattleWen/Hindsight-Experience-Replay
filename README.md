# Dealing with the Sparse Reward Problem in the Bit-Flipping Environment

![](fig/ablation.jpg)

## Getting started
The training environment (PyTorch and dependencies) can be installed as follows:
```bash
cd HER
conda activate -n her python=3.8
pip install -r requirements/requirements.txt
[optional]
pip install -r requirements/requirements-mujoco.txt
...
```

## Train
```
cd ~/slurm
chmod +x run_reward_shaping.sh
./run_reward_shaping.sh
```
