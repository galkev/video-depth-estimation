import numpy as np
import torch
from torch import nn
# from .ddff_net import DDFFEncoderNet, DDFFDecoderNet
from net.rnn_modules import RNNModule, RNNFCEmbeddingModule, RNNStackEmbeddingModule, CCXReduceModule
from tools.tools import obj_to_str
from tools.tools import json_support


@json_support
class AERNN(nn.Module):
    def __init__(self, encoder, decoder, fs_size=10, rnn_dim=2048, rnn_dropout=0,
                 rnn_embedding_type="fc", rnn_type="lstm",
                 rnn_num_layers=1, batch_first=True, ccx_reduce_mode="ccx_last"):
        super().__init__()

        # decoder_in_size = encoder_out_size

        self.encoder = encoder

        print("Encoder encoding dim", encoder.get_out_dim())

        self.rnn_emb_module = self.create_rnn_emb_module(
            encoder.get_out_dim(), rnn_dim, rnn_embedding_type, rnn_dropout,
            rnn_type, batch_first, rnn_num_layers
        )

        self.ccx_reduce = CCXReduceModule(fs_size,
                                          encoder.get_channel_sizes(),
                                          decoder.get_skip_conn_usage(),
                                          mode=ccx_reduce_mode
                                          )

        self.decoder = decoder

    def get_output_mode(self):
        return "last"

    def create_rnn_emb_module(self, encoder_out_size, rnn_dim, rnn_embedding_type, rnn_dropout,
                              rnn_type, batch_first, rnn_num_layers):
        if rnn_embedding_type == "stack":
            rnn_dim = encoder_out_size[1] * encoder_out_size[2]

        rnn_module = RNNModule(rnn_dim,
                               dropout=rnn_dropout,
                               rnn_type=rnn_type,
                               batch_first=batch_first,
                               num_layers=rnn_num_layers
                               )

        if rnn_embedding_type == "fc":
            rnn_emb_module = RNNFCEmbeddingModule(
                rnn_module,
                encoder_out_size,
                # decoder_in_size,
            )
        elif rnn_embedding_type == "stack":
            rnn_emb_module = RNNStackEmbeddingModule(rnn_module)
        else:
            raise Exception(rnn_embedding_type + " not recognized")

        return rnn_emb_module

    def forward(self, x):
        # seq_length = focal stack size
        batch_size, seq_length = x.shape[:2]

        x = x.view(-1, *x.shape[2:])
        x, ccx = self.encoder(x)

        x = self.rnn_emb_module(x, seq_length=seq_length)
        ccx = self.ccx_reduce(batch_size, seq_length, ccx)

        output = self.decoder(x, ccx)

        return output

    def __repr__(self):
        return obj_to_str(self)
