import torch
import wandb
import os
import sys

sys.path.append("/home/compu/cleancode/abc")
from lib import options
from simswap.model import SimSwap


def train(gpu, args): 
    torch.cuda.set_device(gpu)

    model = SimSwap(args, gpu)

    model.initialize_models()
    model.set_dataset()

    if args.use_mGPU:
        model.set_multi_GPU()

    model.set_data_iterator()
    step = model.load_checkpoint()
    model.set_optimizers()
    model.set_loss_collector()

    args.isMaster = gpu == 0

    # Initialize wandb to gather and display loss on dashboard 
    if args.isMaster:
        wandb.init(project=args.project_id, name=args.run_id)

    # Training loop
    global_step = step if step else 0
    while global_step < args.max_step:
        result = model.train_step()

        if args.isMaster:
            # Save and print loss
            if global_step % args.loss_cycle == 0:
                wandb.log(model.loss_collector.loss_dict)
                model.loss_collector.print_loss(global_step)

            # Save image
            if global_step % args.image_cycle == 0:
                model.save_image(result, global_step)

            # Save checkpoint parameters 
            if global_step % args.ckpt_cycle == 0:
                model.save_checkpoint(global_step)

        global_step += 1


if __name__ == "__main__":
    args = options.train_options()
    os.makedirs(args.save_root, exist_ok=True)

    # Set up multi-GPU training
    if args.use_mGPU:  
        args.gpu_num = torch.cuda.device_count()
        args.batch_size = int(args.batch_size / args.gpu_num)
        torch.multiprocessing.spawn(train, nprocs=args.gpu_num, args=(args, ))

    # Set up single GPU training
    else:  
        args.isMaster = True
        train(args.gpu_id, args)
