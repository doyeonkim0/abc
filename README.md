



# Usage

## train

```
python train.py --model faceshifter --run_id simple_test

# if you want to use multi GPUs, add "--use_mGPU"
python train.py --run_id simple_test --use_mGPU

# if you want to use wandb, add "--use_wandb"
python train.py --run_id simple_test --use_mGPU --use_wandb

# if you want to load a checkpoint and retrain it, use "--ckpt_id"
python train.py --run_id simple_test --use_mGPU --use_wandb --ckpt_id={PATH/TO/CKPT} 
 
```
