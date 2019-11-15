import torch

import torchscript
from torchscript_test import random_inputs_torch


def rnn_basic(ins, seq_len, w, b, init_state):
    inputs_time_major = torch.transpose(ins, 1, 0)
    max_seq_len = int(torch.max(seq_len))
    state = init_state
    for i in range(max_seq_len):
        x = inputs_time_major[i]
        h = torch.cat((x, state), 1)
        state = torch.tanh(h @ w + b)  # Basic RNN cell
    return state


if __name__ == "__main__":
    ins, seq_len, w, b, init_state = random_inputs_torch(1, 100, 50, 256)

    converted_fn, _ = torchscript.specialize(rnn_basic, ins, seq_len, w, b, init_state)

    print(converted_fn.graph)

    print(converted_fn(ins, seq_len, w, b, init_state))
