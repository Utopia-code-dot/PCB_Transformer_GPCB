import copy
import json

import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PretrainDataset_one_side(Dataset):
    def __init__(self, data_dir, block_size, device, max_row_col, test_flag=False):
        """
        :param data_dir: str, The path to the data set
        """
        self.data_dir = data_dir
        self.max_row_col = max_row_col
        self.block_size = block_size
        self.device = device
        self.data_info = self.get_data_info()
        self.test_flag = test_flag

    def __getitem__(self, index):  # index returns the specific data required for training
        pairs = self.data_info[index]["pairs"]
        caps = self.data_info[index]["caps"]
        flows = self.data_info[index]["flows"]
        row_col = self.data_info[index]["ori_row_col"]
        pos_cap = self.data_info[index]["pos_cap"]
        pos_flow = self.data_info[index]["pos_flow"]
        pos_pair = self.data_info[index]["pos_pair"]
        case_idx = self.data_info[index]["case_idx"]
        inp_dec = torch.cat((pairs, flows), dim=0)
        inp_enc = caps
        pos_dec = torch.cat((pos_pair, pos_flow), dim=0)
        pos_enc = pos_cap
        x = inp_dec[:-1]
        y = inp_dec[1:]

        # device_type = 'cuda' if "cuda" in self.device else 'cpu'
        # if device_type == 'cuda':
        #     # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        #     x = x.pin_memory().to(self.device, non_blocking=True)
        #     pairs = pairs.pin_memory().to(self.device, non_blocking=True)
        #     cap = cap.pin_memory().to(self.device, non_blocking=True)
        #     row_col = row_col.pin_memory().to(self.device, non_blocking=True)
        #     y = y.pin_memory().to(self.device, non_blocking=True)
        #     table = table.pin_memory().to(self.device, non_blocking=True)
        # else:
        #     x = x.to(self.device)
        #     pairs = pairs.to(self.device)
        #     cap = cap.to(self.device)
        #     row_col = row_col.to(self.device)
        #     y = y.to(self.device)
        #     table = table.to(self.device)

        if self.test_flag:
            return inp_dec, inp_enc, row_col, pos_enc, pos_dec, case_idx
        else:
            return x, inp_enc, row_col, pos_enc, pos_dec, y

    def __len__(self):
        return len(self.data_info)

    def get_data_info(self):
        with open(self.data_dir, "rb") as f:
            data = pickle.load(f)
            f.close()
        with open("encode_table/flow_encode_table.json", "r") as f:
            encode_table = json.load(f)
            f.close()
        with open("./encode_table/start_end_size.pkl", 'rb') as fp:
            start_end_size = pickle.load(fp)
            fp.close()
        # Special codes, used to fill and split, are empty in location codes
        pad, sep, sos, eos, sops, eops, sofs, eofs, socs, eocs = (int(encode_table["<PAD>"]), int(encode_table["<SEP>"]),
                                                                  int(encode_table["<SOS>"]), int(encode_table["<EOS>"]),
                                                                  int(encode_table["<SOPS>"]), int(encode_table["<EOPS>"]),
                                                                  int(encode_table["<SOFS>"]), int(encode_table["<EOFS>"]),
                                                                  int(encode_table["<SOCS>"]), int(encode_table["<EOCS>"]))

        pos_sp_row_enc, pos_sp_col_enc = self.max_row_col[0], self.max_row_col[1]
        data1 = {}

        max_overlap_wires = start_end_size["max_overlap_wires"]
        max_overlap_tiles = start_end_size["max_overlap_tiles"]

        progress_bar = tqdm(data["pairs"].keys(), desc=f'Data processing', unit='sample', position=0, ncols=80)
        for j, i in enumerate(progress_bar):
            ori_row_col = data["ori_row_col"][i]
            case_len = ori_row_col[0] * ori_row_col[1]
            pairs_explain_len = len(data["pairs"][i]) - case_len
            # ---- encode ----
            pairs = data["pairs"][i]
            pairs.insert(case_len, sep)
            pairs.insert(0, sops)
            pairs.append(eops)

            flows = data["flows"][i]
            flows.insert(0, sofs)
            flows.append(eops)

            caps = data["caps"][i]
            caps.insert(0, socs)
            caps.append(eocs)

            # ---- pos encode ----
            row_col_for_each_tile = data["row_col_for_each_tile"][i]
            pos_sp_enc = [pos_sp_row_enc, pos_sp_col_enc]

            pos_cap = [pos_sp_enc]          # SOCS
            pos_cap.extend(row_col_for_each_tile)
            pos_cap.append(pos_sp_enc)      # EOCS

            pos_flow = copy.deepcopy(pos_cap)
            # Interpretation coding of pairs
            pos_pair = [pos_sp_enc]         # SOPS
            pos_pair.extend(row_col_for_each_tile)
            pos_pair.append(pos_sp_enc)     # SEP
            pos_pair.extend([pos_sp_enc for _ in range(pairs_explain_len)])
            pos_pair.append(pos_sp_enc)     # EOPS

            # ---- padding ----
            len_to_pad = self.max_row_col[0]*self.max_row_col[1] - case_len
            max_len_pairs_explain = max_overlap_tiles*(max_overlap_wires+1)
            len_to_pad_pairs = len_to_pad + (max_len_pairs_explain-pairs_explain_len)

            code_pad = [pad for _ in range(len_to_pad)]
            caps.extend(code_pad)
            flows.extend(code_pad)
            pairs.extend([pad for _ in range(len_to_pad_pairs)])

            pos_pad = [pos_sp_enc for _ in range(len_to_pad)]
            pos_cap.extend(pos_pad)
            pos_flow.extend(pos_pad)
            pos_pair.extend([pos_sp_enc for _ in range(len_to_pad_pairs)])

            pairs = torch.tensor(pairs, dtype=torch.int64)
            caps = torch.tensor(caps, dtype=torch.int64)
            flows = torch.tensor(flows, dtype=torch.int64)
            pos_cap = torch.tensor(pos_cap, dtype=torch.int64)
            pos_flow = torch.tensor(pos_flow, dtype=torch.int64)
            pos_pair = torch.tensor(pos_pair, dtype=torch.int64)
            ori_row_col = torch.tensor(ori_row_col, dtype=torch.int64)

            data1[j] = {"pairs": pairs, "caps": caps, "flows": flows, "ori_row_col": ori_row_col,
                        "pos_cap": pos_cap, "pos_flow": pos_flow,
                        "pos_pair": pos_pair, "case_idx": i}

        return data1

# early-stopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='ckpt.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def show_train(file_path, num_epochs=-1):

    show_path = file_path + "/show_dict.pkl"
    # load data
    with open(show_path, 'rb') as file:
        show_dict = pickle.load(file)
    if num_epochs <= 0:
        num_epochs = show_dict["epoch_num"]+1
    legend = ["train loss", "train acc"]
    if "valid_acc" in show_dict:
        legend.append("valid loss")
        legend.append("valid acc")

        valid_acc = show_dict["valid_acc"]
        valid_x = show_dict["valid_x"]
        valid_acc_max_idx = valid_acc.index(max(valid_acc))
        print(f"max accuracy of valid data： {valid_acc[valid_acc_max_idx] * 100:.4f}, Epoch {valid_x[valid_acc_max_idx]}")
    plt.subplot(2, 2, 1)
    plt.plot(show_dict["train_x"], show_dict["train_loss"], label=legend[0])
    plt.title("Training Loss")
    plt.xlabel("epoch")
    plt.xlim(1, num_epochs)

    plt.subplot(2, 2, 2)
    plt.plot(show_dict["train_x"], show_dict["train_acc"], label=legend[1])
    plt.title("Training Accuracy")
    plt.xlabel("epoch")
    plt.xlim(1, num_epochs)

    plt.subplot(2, 2, 3)
    plt.plot(show_dict["valid_x"], show_dict["valid_loss"], label=legend[2])
    plt.title("Validating Loss")
    plt.xlabel("epoch")
    plt.xlim(1, num_epochs)

    plt.subplot(2, 2, 4)
    plt.plot(show_dict["valid_x"], show_dict["valid_acc"], label=legend[3])
    plt.title("Validating  Accuracy")
    plt.xlabel("epoch")
    plt.xlim(1, num_epochs)

    fig, ax = plt.subplots()
    ax.plot(show_dict["train_x"], show_dict["lr"], label="lr")
    ax.set_title("Learning Rate")
    ax.set_xlabel("epoch")
    plt.xlim(1, num_epochs)

    # # plt.suptitle(f"batch_size={batch_size} num_epochs={num_epochs}")
    plt.show()


