"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example（run in terminal）:
$ python train.py --batch_size=16 --compile=False

To run with DDP on 2 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=2 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
import sys

import torch
from torch import nn
import math
from torch.utils.data import DataLoader, random_split
from model import GPT, GPTConfig
from utils import *
from tqdm import tqdm
import pickle
import os
import json
import time
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datetime import datetime

"""-------------------------------------Adjustable parameter-------------------------------------------------------------"""
# ---- model ----
if not os.path.exists(f"./out_train"):
    os.makedirs(f"./out_train")
init_from = "scratch"  # 'scratch'，'resume'
resume_dir = "./out_train/model_20240704_1555"
max_row_col = [50, 50]
n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
compile = True  # use PyTorch 2.0 to compile the model to be faster
num_workers = 0  # num_workers of dataloader
# adamw optimizer
max_lr = 1e-3  # max learning rate
wd = 1e-1  # weight_decay
beta2 = 0.99
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
beta1 = 0.9

# ---- training ----
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
lr_unit = "epoch"
warmup_units = 10  # max warm up iters of lr
lr_decay_units = 100
min_lr = 1e-4  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# 其他参数
gradient_accumulation_steps = 8 * 2  # simulate lager batch_size
device, batch_size = "cuda", 32
device_type = 'cuda' if "cuda" in device else 'cpu'
num_epochs = 100
eval_only = False  # if True, script exits right after the first eval
eval_interval = 1000
eval_iters = 10
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
backend = 'nccl'  # 'nccl', 'gloo', etc. DDP settings

# cal block size
with open("./encode_table/start_end_size.pkl", "rb") as fp:
    tmp_data = pickle.load(fp)
    fp.close()
max_overlap_wires = tmp_data["max_overlap_wires"]
max_overlap_tiles = tmp_data["max_overlap_tiles"]
len_flow = len_pair = max_row_col[0] * max_row_col[1] + 2
len_pair += (max_overlap_wires+1) * max_overlap_tiles + 1
block_size = len_flow + len_pair - 1   # pair+flow

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias,
                  vocab_size=None, cap_encode_size=None, starts_ends_size=None,
                  dropout=dropout)

"""-----------------------------Distributed configuration and command line configuration--------------------------------------------------"""
# systems settings
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

# Distributed configuration
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    print("DDP model!")
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    print(f"ddp_rank:{ddp_rank}, ddp_local_rank:{ddp_local_rank}, ddp_world_size:{ddp_world_size} ")
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    print("DDP False!")
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

now = datetime.now()
out_dir = now.strftime("%Y%m%d_%H%M")
out_dir = f"./out_train/model_{out_dir}"
if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

"""---------------------load data-----------------------"""
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

seed = 1337
torch.manual_seed(seed + seed_offset)
np.random.seed(seed + seed_offset)
torch.cuda.manual_seed(seed + seed_offset)
print("Process the data... (takes a few minutes)")
data_dir = "./preprocess_data/train_data.pkl"
dataset = PretrainDataset_one_side(data_dir, block_size, device, max_row_col)
valid_ratio = 0.1
valid_size = int(valid_ratio * len(dataset))
train_size = len(dataset) - valid_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,
                        num_workers=num_workers)
valid_iter = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True,
                        num_workers=num_workers)


"""------------------------------------function----------------------------------------------"""
def init_model():
    if init_from == 'scratch':
        global model_args
        with open('./encode_table/flow_encode_table.json', 'r') as f:
            content = f.read()
            flow_encode_table = json.loads(content)
            f.close()
            meta_vocab_size = len(flow_encode_table)
            print(f"found vocab_size = {meta_vocab_size} ")
        with open('./encode_table/cap_encode_table.json', 'r') as f:
            content = f.read()
            cap_encode_table = json.loads(content)
            f.close()
            meta_cap_encode_size = len(cap_encode_table)
            print(f"found cap_encode_size = {meta_cap_encode_size} ")
        with open('./encode_table/start_end_size.pkl', 'rb') as f:
            start_end_size = pickle.load(f)
            meta_starts_ends_size = start_end_size['start_end_size']
            print(f"found starts_ends_size = {meta_starts_ends_size} ")

        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        model_args['cap_encode_size'] = meta_cap_encode_size if meta_cap_encode_size is not None else 50304
        model_args['starts_ends_size'] = meta_starts_ends_size if meta_starts_ends_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf, max_row_col, len_pair)
        model.to(device)
        return model, None
    elif init_from == 'resume':
        print(f"Resuming training from {resume_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(resume_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        # create the model
        model_args = checkpoint['model_args']
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf, max_row_col, len_pair)
        state_dict = checkpoint['model']

        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.to(device)
        return model, checkpoint


def get_lr(it_num, epoch_num):
    """
    it_num: the number of batches
    """
    if lr_unit == "iter":
        calc_lr_num = it_num
    elif lr_unit == "step":
        calc_lr_num = it_num // gradient_accumulation_steps
    elif lr_unit == "epoch":
        calc_lr_num = epoch_num
    else:
        sys.exit(f"Learning rate update unit selection error！no {lr_unit} unit！")
    # 1) linear warmup for warmup_steps steps
    if calc_lr_num < warmup_units:
        return max_lr * calc_lr_num / warmup_units
    # 2) if it > lr_decay_steps, return min learning rate
    if calc_lr_num > lr_decay_units:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (calc_lr_num - warmup_units) / (lr_decay_units - warmup_units)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)


def cal_acc_loss(logits, y):
    """
        cal loss and acc
    """
    pred = logits.view(-1, logits.size(-1))
    y = y[:, len_pair:]
    tgt = y.reshape(-1)
    prob = F.softmax(pred, dim=1)

    weights = torch.ones(len(prob[0]), device=device)
    weights[11] = 0.5
    loss = F.cross_entropy(pred, tgt, weight=weights, ignore_index=-1)

    if device_type == "cuda":
        tgt, prob = tgt.cpu(), prob.cpu()
    count = 0
    count_1 = 0
    for i in range(len(tgt)):
        max_pos = prob[i].argmax().item()
        if tgt[i].item() in list(range(10)):
            if max_pos == tgt[i].item():
                continue
            else:
                count = count + 1
            tgt[i] = -1
        else:
            if max_pos == tgt[i].item():
                count = count + 1
                count_1 = count_1 + 1
            else:
                count = count + 1
    accuracy = count_1 / count
    return accuracy, loss


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss_acc(net, data_iter):
    device_type = 'cuda' if "cuda" in device else 'cpu'  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    net.eval()
    metric = Accumulator(3)  # An accumulator for calculating average losses and accuracy
    for i, (X, IE, RC, PE, PD, Y) in enumerate(data_iter):
        with ctx:
            X, IE, RC, PE, PD, Y = X.to(device), IE.to(device), RC.to(device), PE.to(device), PD.to(device), Y.to(
                device)
            logits = net(X, IE, RC, PE, PD)
            accuracy, loss = cal_acc_loss(logits, Y)
        metric.add(loss.item(), accuracy, 1)
    average_loss = metric[0] / metric[2]
    average_acc = metric[1] / metric[2]
    net.train()
    return average_loss, average_acc


def run_train(net, train_iter, valid_iter, checkpoint=None):
    """
    :param net: model
    :param train_iter:
    :param valid_iter:
    :return:
    """
    device_type = 'cuda' if "cuda" in device else 'cpu'  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizer = net.configure_optimizers(wd, max_lr, (beta1, beta2), device_type)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    num_batches, timer = len(train_iter), Timer()

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = net
        net = torch.compile(net)  # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        net = DDP(net, device_ids=[ddp_local_rank])

    if checkpoint is not None:
        file_path = f"{resume_dir}/show_dict.pkl"
        with open(file_path, "rb") as fp:
            show_dict = pickle.load(fp)
        file_path = f"{out_dir}/show_dict.pkl"
        with open(file_path, 'wb') as file:
            pickle.dump(show_dict, file)
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        iter_num = show_dict['iter_num']
        epoch_num = show_dict["epoch_num"]
        best_model = checkpoint['best_model']
        best_val_loss = checkpoint['best_val_loss']
        best_val_acc = checkpoint['best_val_acc']
        best_val_epoch = checkpoint['best_val_epoch']
        checkpoint = None  # free up memory
    else:
        iter_num = 0
        epoch_num = -1
        best_model = net.module.state_dict() if ddp else net.state_dict()
        best_val_acc, best_val_loss, best_val_epoch = -1.0, float("inf"), -1
        show_dict = {"iter_num": 0, "epoch_num": 0,
                     "train_x": [], "train_loss": [], "train_acc": [],
                     "valid_x": [], "valid_loss": [], "valid_acc": [], "time": 0.0,
                     "lr": []}
    running_mfu = -1.0
    raw_net = net.module if ddp else net  # unwrap DDP container
    last_batches_len = num_batches % gradient_accumulation_steps  # Quantity of the last batch
    print(f"saving checkpoint to {out_dir}")

    # loop
    for epoch in range(epoch_num + 1, num_epochs):
        net.train()
        metric = Accumulator(3)

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, epoch) if decay_lr else max_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if master_process:
            progress_bar = tqdm(train_iter, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch', position=0)
        else:
            progress_bar = train_iter

        # progress_bar = train_iter

        # time_3 = time.time()
        for i, (X, IE, RC, PE, PD, Y) in enumerate(progress_bar):
            X, IE, RC, PE, PD, Y = X.to(device), IE.to(device), RC.to(device), PE.to(device), PD.to(device), Y.to(
                device)

            timer.start()
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                if last_batches_len != 0 and i >= num_batches - last_batches_len:
                    net.require_backward_grad_sync = (i == num_batches - 1)
                else:
                    net.require_backward_grad_sync = (
                            i % gradient_accumulation_steps == gradient_accumulation_steps - 1)

            # Obtain the prediction results and calculate the accuracy
            with ctx:
                logits = net(X, IE, RC, PE, PD)
                accuracy, loss = cal_acc_loss(logits, Y)

                if last_batches_len != 0 and i >= num_batches - last_batches_len:
                    loss = loss / last_batches_len
                else:
                    loss = loss / gradient_accumulation_steps

            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

            # Gradient cumulative updating
            if (i == num_batches - 1) or (i % gradient_accumulation_steps == gradient_accumulation_steps - 1):
                # clip the gradient
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)

                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)


            dt = timer.stop()  # delta_time
            mfu = raw_net.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

            metric.add(loss.item(), accuracy, 1)

            if master_process:
                progress_bar.set_postfix(iter=iter_num, train_loss=f"{(metric[0] / metric[2]):.4f}",
                                         train_acc=f"{(metric[1] / metric[2]):.4f}", mfu=f"{running_mfu * 100:.2f}%")

            if ((i + 1) % (num_batches // 5) == 0 or i == num_batches - 1) and master_process:
                show_dict["train_x"].append(epoch + 1 + (i + 1) / num_batches)
                show_dict["train_loss"].append(metric[0] / metric[2])
                show_dict["train_acc"].append(metric[1] / metric[2])
                show_dict["lr"].append(optimizer.param_groups[0]['lr'])
            iter_num += 1
            show_dict["time"] += dt

        # if valid_iter is not None and iter_num % eval_interval == 0 and master_process:
        if valid_iter is not None and master_process:
            raw_net = net.module if ddp else net  # unwrap DDP container
            progress_bar1 = tqdm(valid_iter, desc="Evaluating……", unit="batch", position=1)
            valid_loss, valid_acc = estimate_loss_acc(net, progress_bar1)
            progress_bar1.close()
            show_dict["valid_x"].append(epoch + 1)
            show_dict["valid_acc"].append(valid_acc)
            show_dict["valid_loss"].append(valid_loss)
            show_dict["epoch_num"] = epoch
            show_dict["iter_num"] = iter_num

            file_path = f"{out_dir}/show_dict.pkl"
            with open(file_path, 'wb') as file:
                pickle.dump(show_dict, file)
                file.close()

            if best_val_acc < valid_acc:
                best_model = raw_net.state_dict()
                best_val_acc = valid_acc
                best_val_loss = valid_loss
                best_val_epoch = epoch
            checkpoint = {
                'model': raw_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                "epoch_num": epoch,
                'best_model': best_model,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'best_val_epoch': best_val_epoch,
                'config': config
            }
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            # print(f"\nepoch:{epoch}, iter:{iter_num}, best_valid_acc:{best_val_acc:.4}")
            if master_process:
                tqdm.write(f"\nEpoch: {epoch + 1}, iter: {iter_num}, best_valid_acc: {best_val_acc:.4f}")

    if ddp:
        destroy_process_group()


def train():

    model, checkpoint = init_model()

    # train
    run_train(model, train_iter, valid_iter, checkpoint)

    return out_dir
