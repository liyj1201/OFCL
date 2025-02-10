import argparse
import datetime
import os
import random
import sys
import time

import numpy as np
import torch
from timm.models import create_model
import torch.backends.cudnn as cudnn
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from dataloader import dataloader
import models
import warnings
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch, gc
gc.collect()
torch.cuda.empty_cache()
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')
torch.set_printoptions(threshold=np.inf)

def main(args):
    torch.cuda.empty_cache()
    if args.open:
        from engine_open import train_and_evaluate, evaluate_till_now
    else:
        from engine import train_and_evaluate, evaluate_till_now

    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    data_loader, class_mask = dataloader(args)

    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.n_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.n_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.prompt_length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.pool_size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        ways=args.ways,
        num_tasks=args.n_tasks,
    )
    original_model.to(device)
    model.to(device)


    if args.freeze:
        # frozen all parameters from vit
        for p in original_model.parameters():
            p.requires_grad = False
        # frozen part parameters from vit
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    print(args)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, original_model, data_loader, device,
                                            task_id, class_mask, acc_matrix, args,)

        return

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0

    optimizer = create_optimizer(args, model)

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.base_epochs} epochs in session 0 and {args.epochs} epochs in new session")
    start_time = time.time()
    train_and_evaluate(model, original_model,
                       criterion, data_loader, optimizer, lr_scheduler,
                       device, class_mask, args)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    from configs import get_args_parser
    parser = argparse.ArgumentParser('OwFSCIL configs')
    get_args_parser(parser)
    args = parser.parse_args()
    main(args)
    sys.exit(0)