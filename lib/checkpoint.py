import torch
import os

        
def load_checkpoint(args, model, name, global_step=-1):
    if global_step == -1:
        idx = 'latest'
    else:
        idx = global_step
    ckpt_path = f'{args.save_root}/{args.ckpt_id}/ckpt/{name}_{idx}.pt'
    ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))
    model.load_state_dict(ckpt, strict=False)


def save_checkpoint(args, model, global_step, name):
    dir_path = f'{args.save_root}/{args.run_id}/ckpt'
    os.makedirs(dir_path, exist_ok=True)
    
    ckpt_path = dir_path + f'/{name}_{global_step}.pt'
    torch.save(model.state_dict(), ckpt_path)

    latest_ckpt_path = dir_path + f'/{name}_latest.pt'
    torch.save(model.state_dict(), latest_ckpt_path)
        