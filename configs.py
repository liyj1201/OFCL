
def get_args_parser(subparsers):
    # model parameters
    subparsers.add_argument('--input_size', default=224, type=int, help='images input size')
    subparsers.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                            help='Name of model to train')
    subparsers.add_argument('--pretrained', default=True, help='Load pretrained model or not')
    subparsers.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate')
    subparsers.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate')

    # few shot class incremental learning parameters
    subparsers.add_argument('--n_tasks', default=9, type=int, help='number of tasks')
    subparsers.add_argument('--n_classes', default=100, type=int, help='number of classes')
    subparsers.add_argument('--ways', default=5, type=int, help='number of classes per task')
    subparsers.add_argument('--base_ways', default=60, type=int, help='number of classes in session 0')
    subparsers.add_argument('--shots', default=5, type=int, help='number of training samples per way')
    subparsers.add_argument('--task_inc', default=False, type=bool, help='if doing task incremental')

    # training and test parameters
    subparsers.add_argument('--dataset', default='MiniImagenet', type=str, help='dataset name')
    subparsers.add_argument('--num_workers', default=0, type=int)
    subparsers.add_argument('--batch_size', default=25, type=int, help='Batch size per device')
    subparsers.add_argument('--pin-mem', action='store_true',
                            help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    subparsers.add_argument('--device', default='cuda', help='device to use for training / testing')
    subparsers.add_argument('--seed', default=42, type=int)
    subparsers.add_argument('--data_path', default='./local_datasets/', type=str, help='dataset path')
    subparsers.add_argument('--epochs', default=200, type=int)
    subparsers.add_argument('--base_epochs', default=5, type=int)
    subparsers.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
    subparsers.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
    subparsers.add_argument('--eval', action='store_true', help='Perform evaluation only')

    # Optimizer parameters
    subparsers.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    subparsers.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                            help='Optimizer Epsilon (default: 1e-8)')
    subparsers.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA',
                            help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    subparsers.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',
                            help='Clip gradient norm (default: None, no clipping)')
    subparsers.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    subparsers.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    subparsers.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')

    # Learning rate schedule parameters
    subparsers.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER',
                            help='LR scheduler (default: "constant"')
    subparsers.add_argument('--lr', type=float, default=0.03, metavar='LR', help='learning rate (default: 0.03)')
    subparsers.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                            help='learning rate noise on/off epoch percentages')
    subparsers.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                            help='learning rate noise limit percent (default: 0.67)')
    subparsers.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                            help='learning rate noise std-dev (default: 1.0)')
    subparsers.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                            help='warmup learning rate (default: 1e-6)')
    subparsers.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                            help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    subparsers.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    subparsers.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                            help='epochs to warmup LR, if scheduler supports')
    subparsers.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                            help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    subparsers.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                            help='patience epochs for Plateau LR scheduler (default: 10')
    subparsers.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                            help='LR decay rate (default: 0.1)')
    subparsers.add_argument('--unscale_lr', type=bool, default=True, help='scaling lr by batch size (default: True)')
    # Misc parameters
    subparsers.add_argument('--print_freq', type=int, default=10, help='The frequency of printing')

    # prompt learning parameters
    subparsers.add_argument('--prompt_length', default=5, type=int, )
    subparsers.add_argument('--embedding_key', default='cls', type=str)
    subparsers.add_argument('--prompt_key_init', default='uniform', type=str)
    subparsers.add_argument('--prompt_pool', default=True, type=bool, )
    subparsers.add_argument('--prompt_key', default=True, type=bool, )
    subparsers.add_argument('--pool_size', default=25, type=int, )
    subparsers.add_argument('--top_k', default=20, type=int, )
    subparsers.add_argument('--batchwise_prompt', default=True, type=bool)
    subparsers.add_argument('--use_prompt_mask', default=False, type=bool)
    subparsers.add_argument('--pull_constraint', default=True)
    subparsers.add_argument('--pull_constraint_coeff', default=1, type=float)

    # Vit parameters
    subparsers.add_argument('--head_type', default='prompt', choices=['token', 'gap', 'prompt', 'token+prompt'],
                            type=str, help='input type of classification head')
    subparsers.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*',
                            type=list, help='freeze part in backbone model')

    # open detection
    subparsers.add_argument('--R', default=3, type=float,help='Initial radius for each class')
    subparsers.add_argument('--M', default=1, type=float, help='Margin')
    subparsers.add_argument('--Lambda', default=0.1, type=float, help='')
    subparsers.add_argument('--Alpha', default=1, type=float, help='')
    subparsers.add_argument('--Beta', default=3, type=float, help='')
    subparsers.add_argument('--quan', default=0.1, type=float, help='')
    subparsers.add_argument('--open', default=True, type=bool, help='open or not')
    subparsers.add_argument('--epsilon', default=3.5, type=float, help='Unknown classes radius')
    subparsers.add_argument('--min_samples', default=5, type=float, help='Minimum number of samples in clustering')
