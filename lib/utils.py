import torch
import torchvision
import cv2
import os


def get_grid_row(images):
    # Make row of 8 images
    images = images[:8]
    grid_row = torchvision.utils.make_grid(images.detach().cpu(), nrow=images.shape[0]) * 0.5 + 0.5
    return grid_row


def save_image(args, global_step, dir, images):
    dir_path = f'{args.save_root}/{args.run_id}/{dir}'
    os.makedirs(dir_path, exist_ok=True)
    
    sample_image = make_grid_images(images).transpose([1,2,0]) * 255
    cv2.imwrite(dir_path + f'/e{global_step}.jpg', sample_image[:,:,::-1])


def make_grid_images(images):
    grid_rows = []

    for image in images:
        grid_row = get_grid_row(image)
        grid_rows.append(grid_row)

    grid = torch.cat(grid_rows, dim=1).numpy()
    return grid


def setup_ddp(gpu, ngpus_per_node):
    torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=ngpus_per_node,
            rank=gpu)


def update_net(optimizer, loss):
    optimizer.zero_grad()  
    loss.backward()   
    optimizer.step()  
