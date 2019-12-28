import torch
from torch import nn
import numpy as np
from tools.tools import obj_to_str, json_support


class FourDirLSTM(nn.Module):
    def __init__(self,
                 dir_length,
                 batch_size):
        super().__init__()

        self.dir_length = dir_length
        self.batch_size = batch_size

        self.rnns = [torch.nn.LSTM(self.dir_length, self.dir_length) for _ in range(4)]
        self.hidden = [(
            torch.zeros([1, self.batch_size, self.dir_length]),
            torch.zeros([1, self.batch_size, self.dir_length]))
            for _ in range(4)]

    def forward(self, x):
        block = x.view(self.batch_size, self.dir_length, 2 * self.dir_length)

        rev_idx = torch.arange(self.dir_length - 1, 0 - 1, -1)

        sub_blocks = [None] * 4

        sub_blocks[1] = block[:, :, :self.dir_length].permute(1, 0, 2)
        sub_blocks[0] = sub_blocks[1].permute(2, 1, 0)
        sub_blocks[3] = block[:, :, self.dir_length:].index_select(1, rev_idx).index_select(2, rev_idx).permute(1, 0, 2)
        sub_blocks[2] = sub_blocks[3].permute(2, 1, 0)

        blocks_out = [None] * 4

        for i, rnn in enumerate(self.rnns):
            out, self.hidden[i] = rnn(sub_blocks[i], self.hidden[i])
            blocks_out[i] = out[-1]

        block_cat = torch.cat(blocks_out, dim=1)

        return block_cat


@json_support
class RNNModule(nn.Module):
    def __init__(self,
                 rnn_input_dim=2048, rnn_hidden_size=None,
                 dropout=0, bias=True, rnn_type="lstm", batch_first=True, num_layers=1):
        super().__init__()

        self.rnn_input_dim = rnn_input_dim
        self.rnn_hidden_size = rnn_hidden_size if rnn_hidden_size is not None else rnn_input_dim

        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.rnn = None
        self._init_rnn(bias, dropout, batch_first)

        self.hidden = None

    def _init_rnn(self, bias, dropout, batch_first):
        params = {
            "input_size": self.rnn_input_dim,
            "hidden_size": self.rnn_hidden_size,
            "num_layers": self.num_layers,
            "bias": bias,
            "dropout": dropout,
            "bidirectional": False,
            "batch_first": batch_first
        }

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(**params)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(**params)
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(**params)

    def init_hidden(self, batch_size):
        # (num_layers * num_directions, batch, hidden_size)
        hidden_size = [self.num_layers, batch_size, self.rnn_hidden_size]

        dev = next(self.parameters()).device

        if self.rnn_type == "lstm":
            self.hidden = (torch.zeros(hidden_size, device=dev),
                           torch.zeros(hidden_size, device=dev))
        else:
            self.hidden = torch.zeros(hidden_size, device=dev)

        print("RNNModule.init_hidden")

    def forward(self, x):
        if not self.rnn.batch_first:
            x = x.permute(1, 0, 2)

        if self.hidden is not None:
            x, self.hidden = self.rnn(x, self.hidden)
        else:
            x, _ = self.rnn(x)  # (batch, seq, feature)

        if not self.rnn.batch_first:
            x = x.permute(1, 0, 2)

        return x

    def __repr__(self):
        return obj_to_str(self)


@json_support
class RNNEmbeddingModuleBase(nn.Module):
    def __init__(self, rnn_module):
        super().__init__()
        self.rnn_module = rnn_module

    def get_rnn(self):
        return self.rnn_module

    def _select_output(self, x, out_select):
        if out_select is not None:
            x = x[:, out_select]

        return x

    def _rnn_out_seq_length(self, in_seq_length, rnn_out_select):
        if rnn_out_select is None:
            return in_seq_length
        if isinstance(rnn_out_select, int):
            return 1
        else:
            return len(rnn_out_select)

    def get_input_params(self, x, seq_length, rnn_out_select):
        batch_size = x.shape[0] // seq_length
        orig_shape = x.shape[1:]
        out_seq_length = self._rnn_out_seq_length(seq_length, rnn_out_select)

        return batch_size, orig_shape, out_seq_length

    def forward(self, *inputs):
        raise NotImplementedError


@json_support
class RNNStackEmbeddingModule(RNNEmbeddingModuleBase):
    def __init__(self, rnn_module):
        super().__init__(rnn_module)

    # x: (batch_size * fs_size, c(feature_maps)=512, h=7, w=7)
    def forward(self, x, seq_length):
        rnn_out_select = -1

        batch_size, orig_shape, out_seq_length = self.get_input_params(x, seq_length, rnn_out_select)
        num_feat_maps = x.shape[1]
        feat_map_size = x.shape[2:]

        # x [B * S, C, H, W]
        x = x.view(batch_size, seq_length * num_feat_maps, np.prod(feat_map_size))  # stack feature maps of all inputs
        # x [B, S * C, H * W]
        x = self.rnn_module(x)  # run the rnn over all feature maps
        # x [B, S * C, H * W]
        x = x.view(batch_size, seq_length, num_feat_maps, *feat_map_size)  # split feature maps of inputs again
        # x [B, S, C, H, W]
        x = self._select_output(x, rnn_out_select)  # select desired output (usually last one)
        # x [B, C, H, W]
        return x


@json_support
class RNNFCEmbeddingModule(RNNEmbeddingModuleBase):
    def __init__(self,
                 rnn_module,
                 input_dim,
                 output_dim=None):
        super().__init__(rnn_module)

        if output_dim is None:
            output_dim = input_dim

        self.fc1 = nn.Linear(np.prod(input_dim), rnn_module.rnn_input_dim)
        self.fc2 = nn.Linear(rnn_module.rnn_hidden_size, np.prod(output_dim))

    # x: (batch_size * fs_size, c(feature_maps)=512, h=7, w=7)
    def forward(self, x, seq_length):
        rnn_out_select = -1

        batch_size, orig_shape, out_seq_length = self.get_input_params(x, seq_length, rnn_out_select)

        x = x.view(batch_size * seq_length, np.prod(orig_shape))  # flatten x[1:]
        x = self.fc1(x)
        x = x.view(batch_size, seq_length, self.rnn_module.rnn_input_dim)  # split batch and seq dimension

        x = self.rnn_module(x)  # batch first
        x = self._select_output(x, rnn_out_select)  # select frames (usually last)

        # x = x.view(batch_size * out_seq_length, self.rnn_module.rnn_hidden_size)  # combine batch and seq dim again
        x = self.fc2(x)
        x = x.view(batch_size * out_seq_length, *orig_shape)  # restore x[1:] original shape

        return x


@json_support
class CCXReduceModule(nn.Module):
    # ccx
    # torch.Size([2, 64, 224, 224])
    # torch.Size([2, 128, 112, 112])
    # torch.Size([2, 256, 56, 56])
    # torch.Size([2, 512, 28, 28])
    # torch.Size([2, 512, 14, 14])
    # ddff_ccx_num_channels = [64, 128, 256, 512, 512]

    def __init__(self, fs_size, channel_sizes, ccx_activated, mode="ccx_last"):
        super().__init__()

        self.ccx_activated = ccx_activated
        self.mode = mode

        if self.mode == "ccx_conv":
            self.conv = torch.nn.ModuleList([
                nn.Conv2d(channel_sizes[i]*fs_size, channel_sizes[i], 1, bias=False)
                if ccx_activated[i] else None
                for i in range(len(self.ccx_activated))
            ])

    def process_ccx(self, batch_size, fs_size, i, ccx):
        if self.mode == "ccx_last":
            return ccx.view(batch_size, fs_size, *ccx.shape[1:])[:, -1]
        elif self.mode == "ccx_conv":
            return self.conv[i](ccx.view(batch_size, -1, *ccx.shape[2:]))

    def forward(self, batch_size, fs_size, ccx):
        return [self.process_ccx(batch_size, fs_size, i, ccx[i]) if self.ccx_activated[i] else ccx[i] for i in range(len(ccx))]