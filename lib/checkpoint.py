import torch
import os

        
def load_checkpoint(args, model, optimizer, name, global_step=-1):    
    if global_step == -1:
        idx = 'latest'
    else:
        idx = global_step

    ckpt_path = f'{args.save_root}/{args.ckpt_id}/ckpt/{name}_{idx}.pt'

    try:
        ckpt_dict = torch.load(ckpt_path, map_location=torch.device('cuda'))
        model.load_state_dict(ckpt_dict['model'], strict=False)
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        return ckpt_dict['step']
    except:
        if args.isMaster:
            print(f"Failed to load checkpoint of {name}.")
        return 0


def save_checkpoint(args, model, optimizer, name, global_step):
    ckpt_dict = {}
    ckpt_dict['model'] = model.state_dict()
    ckpt_dict['optimizer'] = optimizer.state_dict()
    ckpt_dict['step'] = global_step

    dir_path = f'{args.save_root}/{args.run_id}/ckpt'
    os.makedirs(dir_path, exist_ok=True)
    
    ckpt_path = dir_path + f'/{name}_{global_step}.pt'
    torch.save(ckpt_dict, ckpt_path)

    latest_ckpt_path = dir_path + f'/{name}_latest.pt'
    torch.save(ckpt_dict, latest_ckpt_path)
        