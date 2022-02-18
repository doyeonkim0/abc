import argparse


def train_options():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model', type=str, required=True) 

    # Experiment id
    parser.add_argument('--run_id', type=str, required=True) 
    parser.add_argument('--gpu_id', type=int, default=0) 
    parser.add_argument('--ckpt_id', type=str, default=None)

    # Hyperparameters
    parser.add_argument('--batch_per_gpu', type=str, default=8)
    parser.add_argument('--max_step', type=str, default=300000)
    parser.add_argument('--same_prob', type=float, default=0.2)

    # Dataset
    parser.add_argument('--dataset_root_list', type=list, \
        # default=['/home/compu/dataset/kface_wild_cv2_256'])
        default=['/home/compu/dataset/CelebHQ'])
        # default=['/home/compu/dataset/kface_wild_1024'])
        # default=['/home/compu/datasets/k-celeb'])

    # Learning rate
    parser.add_argument('--lr_G', type=str, default=1e-4)
    parser.add_argument('--lr_D', type=str, default=1e-5)

    # Log
    parser.add_argument('--loss_cycle', type=str, default=10)
    parser.add_argument('--image_cycle', type=str, default=1000)
    parser.add_argument('--ckpt_cycle', type=str, default=10000)
    parser.add_argument('--save_root', type=str, default="training_result")

    # Weight
    parser.add_argument('--W_id', type=float, default=5)
    parser.add_argument('--W_shape', type=float, default=0)
    parser.add_argument('--W_adv', type=float, default=1)
    parser.add_argument('--W_recon', type=float, default=10)
    parser.add_argument('--W_seg', type=float, default=0)
    parser.add_argument('--W_cycle', type=float, default=1)
    parser.add_argument('--W_lpips', type=float, default=0)
    parser.add_argument('--W_attr', type=float, default=10)
    parser.add_argument('--W_fm', type=float, default=0)

    # Multi GPU
    parser.add_argument('--isMaster', default=False)
    parser.add_argument('--use_mGPU', action='store_true')

    # Use wandb
    parser.add_argument('--use_wandb', action='store_true')

    return parser.parse_args()
