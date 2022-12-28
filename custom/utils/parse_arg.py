
class DefaultArgs:
    batch_size = 50
    epochs = 1
    accum_iter = 1
    model = 'efficientnet_b1'
    input_size = 200
    crop_size = 170
    drop_path = 0.1
    clip_grad = None
    weight_decay = 0
    lr = 0.01
    layer_decay = 0.75
    min_lr = 0.01
    warmup_epochs = 0
    color_jitter = None # 'Color jitter factor (enabled only when not using Auto/RandAug)'
    aa = None # 'Use AutoAugment policy. "v0" or "original". " + "(eg. rand-m9-mstd0.5-inc1)'
    smoothing = 0 # 'Label smoothing (default: 0.1)'
    reprob = 0
    remode = None # 'Random erase mode (default: "pixel")'
    recount = 0
    resplit = False
    mixup = 0
    cutmix = 0.
    cutmix_minmax = None
    mixup_prob = 0.
    mixup_switch_prob = 0.
    mixup_mode = 'batch'
    finetune = ''
    global_pool = False
    cls_token = True
    data_path = ''
    anno_path = ''
    nb_classes = 2
    output_dir = ''
    log_dir = 'tensorboard_logs'
    device = 'cpu'
    seed = 666
    resume = ''
    start_epoch = 0
    eval = False
    dist_eval = False
    num_workers = 4
    pin_mem = True
    no_pin_mem = False
    world_size = 1
    local_rank = -1
    dist_on_itp = False
    dist_url = 'env://'
    label = ''
    heterogeneous = True
    other_feature_size = 1
    hidden_size = 256
    combine_method = 'concat'
        
def parse_args(args):
    final_args = DefaultArgs()
    
    if len(args) == 0:
        return final_args
    
    args = args.strip().split(' ')
    for arg_val in args:
        arg, val = arg_val.split('=')
        try:
            val = eval(val)
        except Exception as e:
            # val is str
            pass
        setattr(final_args, arg, val)
    return final_args

