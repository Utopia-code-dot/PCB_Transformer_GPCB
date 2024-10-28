import os
import sys
import time
import json
import copy
import pickle
from contextlib import nullcontext
import torch
import random
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from envs.FlowSolverPlg import FlowSolver
from model import GPTConfig, GPT
from utils import PretrainDataset_one_side, Timer, Accumulator
import pickle
from tqdm import tqdm

"""-------------------------------------Adjustable parameter-------------------------------------------------------------"""
# -----------------------------------------------------------------------------
data_dir = "./preprocess_data/test_data.pkl"
sample_num = -1       # the number of sample to test,if =-1 test all samples
temperature = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = None  # retain only the top_k most likely tokens, clamp others to have 0 probability, None represent maximum probability
max_row_col = [50, 50]  # max row and col of block
sample_times = 1
seed = 1337
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster

# cal block size
with open("./encode_table/start_end_size.pkl", "rb") as fp:
    tmp_data = pickle.load(fp)
    fp.close()
max_overlap_wires = tmp_data["max_overlap_wires"]
max_overlap_tiles = tmp_data["max_overlap_tiles"]
len_flow = len_pair = max_row_col[0] * max_row_col[1] + 2
len_pair += (max_overlap_wires+1) * max_overlap_tiles + 1
block_size = len_flow + len_pair - 1   # pair+flow
flow_start_idx = len_pair + 1

exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


"""------------------------------------function----------------------------------------------"""


def init_model(out_dir):
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = {}
    model_args['block_size']: int = 5305
    model_args['vocab_size']: int = 496
    model_args['cap_encode_size']: int = 896
    model_args['starts_ends_size']: int = 140
    model_args['n_layer']: int = 6
    model_args['n_head']: int = 8
    model_args['n_embd']: int = 256
    model_args['dropout']: float = 0.0
    model_args['bias']: bool = False
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, max_row_col, len_pair)
    if "best_model" in checkpoint:
        state_dict = checkpoint['best_model']
    else:
        state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)
    return model


def plot_flows(mat, wid=100, fig_title=None, save_path=None):
    row, col = mat.shape
    draw_mat = np.zeros((row * wid, col * wid))
    # draw tiles
    for r in range(1, row):
        r *= wid
        cv2.line(draw_mat, (0, r), (draw_mat.shape[1] - 1, r), color=150, thickness=3)

    for c in range(1, col):
        c *= wid
        cv2.line(draw_mat, (c, 0), (c, draw_mat.shape[0] - 1), color=150, thickness=3)

    with open(f"./encode_table/flow_encode_table.json", 'rb') as fp:
        flow_encode_table = json.load(fp)
        fp.close()
    flow_decode_table = {v:k for k,v in flow_encode_table.items()}
    for x in range(row):
        for y in range(col):
            tile = (wid * x, wid * y, wid * (x + 1), wid * (y + 1))
            flows = flow_decode_table[str(mat[x][y])]
            FlowSolver(tile=tile, flows=eval(flows)).show_flows(draw_mat)
    # plt.show()
    plt.matshow(draw_mat)
    if fig_title:
        plt.title(fig_title)
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()


def pad_data(data, size):

    pad = ((0, size - len(data)), (0, size - len(data[0])))
    data_pad = np.pad(data, pad, mode="constant", constant_values=0)

    return data_pad


def cal_pred_acc(pred, tgt):
    cmp = pred == tgt
    return (cmp.sum()/len(tgt[0])).item()


def run_sample(model, flow_tgt, ori_x, c_in, pos_encoder, pos_decoder, predict_len, sample_times, temperature=1.0, top_k=None):
    acc_max = 0
    flow_pred_max = None
    dt_for_acc_max = 0
    timer = Timer()
    x_in = ori_x[:, :flow_start_idx]  # <SOFS>

    # forward to calculate k,v
    with ctx:   # Calculate the location of the encoder's k, v and decoder in advance
        k, v = model.calculate_k_v(c_in, pos_encoder)
        pos_emb_dec = model.calc_pos_dec_emb(pos_decoder)

    # begin to generate
    for cnt in range(sample_times):
        timer.start()
        max_new_tokens = tqdm(range(predict_len), desc=f'生成中……', unit="token")
        with torch.no_grad():
            with ctx:
                x_out = model.generate(x_in, k, v, pos_emb_dec, max_new_tokens, temperature, top_k)
        dt = timer.stop()
        # get the flow part
        x_out = x_out[:, flow_start_idx:]
        # calculate accuracy
        acc = cal_pred_acc(x_out, flow_tgt)
        if acc > acc_max:
            acc_max = acc
            flow_pred_max = x_out
            dt_for_acc_max = dt
    print(
        f"Max Prediction Accuracy：{acc_max:.2f}, with time:{dt_for_acc_max:.2f}s")
    return flow_pred_max, acc_max


def sample(out_dir='./out_train/model_20240719_2036'):

    # load model
    sample_model = init_model(out_dir)
    # load data
    dataset = PretrainDataset_one_side(data_dir, block_size, device, max_row_col, test_flag=True)
    if sample_num == -1:
        pass
    else:
        sample_idx = random.sample(range(len(dataset)), sample_num)
        dataset = [dataset[idx] for idx in sample_idx]
    data_iter = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True)
    metric = Accumulator(2)  # for accuracy record
    # sample loop
    for idx, (X, IE, RC, PE, PD, case_idx) in enumerate(data_iter):
        case_idx = case_idx.item()
        row, col = [RC[0][i].item() for i in range(2)]
        env_size = row * col
        predict_len = env_size
        X, IE, PE, PD = X.to(device), IE.to(device), PE.to(device), PD.to(device)
        flow_tgt = X[:, flow_start_idx: flow_start_idx + predict_len]   # Valid flow label
        print(f"sample{idx}, case_idx: {case_idx}, size: {env_size}")
        flow_predict_max, acc = run_sample(sample_model, flow_tgt, X, IE, PE, PD, predict_len, sample_times, temperature, top_k)
        metric.add(acc, 1)
        # Visual display
        flow_pred_mat = flow_predict_max.cpu().numpy().reshape(row, col)
        flow_tgt_mat = flow_tgt.cpu().numpy().reshape(row, col)

        if not os.path.exists("./predict_result"):
            os.makedirs("./predict_result")
        with open(f"./predict_result/case_{case_idx}.pkl", "wb") as fp:
            pickle.dump(flow_pred_mat, fp)
            fp.close()
        fig_title = f"env label"
        plot_flows(flow_tgt_mat, wid=100, fig_title=fig_title)
        fig_title = f"env pred -- acc:{acc:.2f}"
        plot_flows(flow_pred_mat, wid=100, fig_title=fig_title)
        plt.show()
        print("\n")
    print(f"Number of test: {int(metric[1])}")
    print(f"Average total accuracy: {metric[0] / metric[1]:.4f}")

