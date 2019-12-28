import torch
from torch import nn
from net.pool_net import PoolNet


class ConsecNetBase(nn.Module):
    def __init__(self, nets, net_input_indices=None):
        super().__init__()

        assert len(nets) == 2

        self.nets = nn.ModuleList(nets)

        self.net_input_indices = net_input_indices

    def get_output_mode(self, i=None):
        return self.nets[-1].get_output_mode()

    def get_net(self, idx):
        return self.nets[idx]

    # add_input = output from last before
    def _forward_net(self, net_idx, x_input, add_input=None):
        if add_input is None:
            add_input = []

        net = self.nets[net_idx]

        if self.net_input_indices is not None:
            in_indices = self.net_input_indices[net_idx]
        else:
            """
            add_input_dim = sum([x.shape[2] for x in add_input])
            x_input_dim = net.get_in_channels_count() - add_input_dim

            if x_input_dim < 3 or x_input_dim > 4:
                raise Exception("Error check if correct")

            in_indices = range(x_input_dim)
            """
            raise Exception("net_input_indices most be specified now")

        for idx in in_indices:
            assert idx < x_input.shape[2]

        if len(in_indices) > 0:
            net_in = torch.cat([x_input[:, :, in_indices]] + add_input, dim=2)
        else:
            # print(add_input.shape)
            net_in = add_input[0]

        """
        if len(add_input) > 0:
            print(net_in.shape)
            print(net_in)
            raise Exception()
        """

        return net(net_in)


class ConsecNet(ConsecNetBase):
    def __init__(self, last_out_only=False, **kwargs):
        super().__init__(**kwargs)

        self.last_out_only = last_out_only

    def forward(self, x):
        net1_out = self._forward_net(0, x)
        net2_out = self._forward_net(1, x, [net1_out])

        if self.last_out_only:
            return net2_out
        else:
            return net1_out, net2_out


def make_consec_net(nets, net_inputs):
    net_input_indices = []

    for net_input in net_inputs:
        indices = []

        if "c" in net_input:
            indices += [0, 1, 2]

        if "f" in net_input:
            indices += [3]

        net_input_indices.append(indices)

    return ConsecNet(nets=nets, net_input_indices=net_input_indices)


class CoCToDepth(nn.Module):
    def __init__(self, dataset, use_signed_coc=False):
        super().__init__()

        self.use_signed_coc = use_signed_coc

        self.lens = dataset.lens
        self.coc_norm = dataset.signed_coc_normalize if self.use_signed_coc else dataset.coc_normalize
        self.depth_norm = dataset.depth_normalize

    def forward(self, focus_dist, coc, fgbg=None):
        if fgbg is not None:
            return self.lens.get_depth_from_fgbg_coc(
                focus_distance=focus_dist,
                fgbg=fgbg,
                coc=coc,
                coc_normalize=self.coc_norm,
                depth_normalize=self.depth_norm)
        else:
            return self.lens.get_depth_from_signed_coc(
                focus_distance=focus_dist,
                signed_coc=coc,
                signed_coc_normalize=self.coc_norm,
                depth_normalize=self.depth_norm
            )


class PreDepthConsecNet(ConsecNetBase):
    def __init__(self, dataset=None, **kwargs):
        super().__init__(**kwargs)

        self.coc_to_depth = CoCToDepth(dataset) if dataset else None

    def set_dataset(self, dataset):
        self.coc_to_depth = CoCToDepth(dataset)

    def forward(self, x, focus_dist):
        coc = self._forward_net(0, x)
        predepth = self.coc_to_depth(focus_dist, coc)

        depth = self._forward_net(1, x, [coc, predepth])

        return coc, depth
