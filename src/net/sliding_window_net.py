import torch
from torch import nn
from net.modules import AvgReduce
from tools.tools import type_adv


class NetOutputQueue:
    def __init__(self, net, shift, pos, wnd_size, input_seq):
        super().__init__()

        self.net = net

        self.cur_out = None
        # self.shift = shift
        self.cur_wnd_start_idx = shift

        self.cur_pos = pos
        self.input_seq = input_seq

        self.wnd_size = wnd_size

        self._load()

        # print(shift, "popping", pos - shift)
        for _ in range(pos - shift):
            self.cur_out.pop(0)

        # print(shift, "popped", pos -shift)

    def _get_wnd_start(self):
        return self.cur_wnd_start_idx

    def _get_wnd_end(self):
        return self._get_wnd_start() + self.wnd_size

    def _is_loaded(self):
        return self._get_wnd_start() <= self.cur_pos < self._get_wnd_end()

    def _load(self):
        # print(str(self.shift) + "|" + (" " * self._get_wnd_start()) + ("^" * self.wnd_size))
        self.cur_out = list(torch.unbind(self.net(
            self.input_seq[:, self._get_wnd_start():self._get_wnd_end()].contiguous()), dim=1))

    def finished(self):
        return self.cur_pos + self.wnd_size - 1 >= self.input_seq.shape[1]

    def pop(self):
        if not self._is_loaded():
            assert len(self.cur_out) == 0
            # print(str(self.shift) + "|" "Loading", self.cur_pos, "not in ", self._get_wnd_start(), self._get_wnd_end()-1)
            self.cur_wnd_start_idx += self.wnd_size
            self._load()

        self.cur_pos += 1

        # print(str(self.shift) + "| Remaining in buffer " + str(len(self.cur_out) - 1))
        return self.cur_out.pop(0)


class SlidingWindowNoReduceNet(nn.Module):
    def __init__(self, wnd_size, net, stride=1):
        super().__init__()

        self.net = net
        self.wnd_size = wnd_size
        self.stride = stride if self.net.get_output_mode() != "all" else self.wnd_size

    def forward(self, input_seq):
        len_input_seq = input_seq.shape[1]

        outputs_list = None

        for i in range(0, len_input_seq - (self.wnd_size - 1), self.stride):
            x = input_seq[:, i:i+self.wnd_size]
            outputs = self.net(x)

            if not isinstance(outputs, tuple):
                outputs = tuple([outputs])

            if outputs_list is None:
                outputs_list = [[] for _ in range(len(outputs))]

            for out_list, out in zip(outputs_list, outputs):
                if len(out.shape) == 5:
                    if len(outputs[-1].shape) == 5:
                        out = list(torch.unbind(out, dim=1))
                    elif len(outputs[-1].shape) == 4:
                        out_mode = self.net.get_output_mode()
                        if out_mode == "last":
                            out_idx = -1
                        elif out_mode == "middle":
                            out_idx = out.shape[1] // 2
                        else:
                            raise Exception("Error")

                        out = [out[:, out_idx]]
                else:
                    out = [out]

                out_list += out
        for i in range(len(outputs_list)):
            out = outputs_list[i]

            if len(out) < len_input_seq:
                out_mode = self.net.get_output_mode()

                diff = len_input_seq - len(out)
                zero_img = torch.zeros_like(out[0])
                if out_mode == "last":
                    out = [zero_img] * diff + out
                elif out_mode == "middle":
                    out = [zero_img] * (diff // 2) + out + [zero_img] * ((diff + 1) // 2)
                elif out_mode == "all":
                    out = out + [zero_img] * diff

                assert len(out) == len_input_seq

            outputs_list[i] = out

        outputs_list = [torch.stack(l, dim=1) for l in outputs_list]

        if len(outputs_list) == 1:
            return outputs_list[0]
        else:
            return outputs_list


class SlidingWindowNet(nn.Module):
    def __init__(self, wnd_size, net, reduce=AvgReduce(), use_lazy_fwd=False, stride=1):
        super().__init__()

        assert stride == 1

        self.net = net
        self.reduce = reduce
        self.wnd_size = wnd_size
        self.use_lazy_fwd = use_lazy_fwd

    def get_output_mode(self):
        return self.net.get_output_mode()

    def _forward_lazy(self, input_seq):
        output_queues = [
            NetOutputQueue(self.net, shift, pos=self.wnd_size - 1, wnd_size=self.wnd_size, input_seq=input_seq)
            for shift in range(self.wnd_size)
        ]

        out_frames = []

        while not output_queues[0].finished():
            x = torch.stack([queue.pop() for queue in output_queues], dim=1)
            x = self.reduce(x)
            out_frames.append(x)

        out = torch.stack(out_frames, dim=1)
        return out

    def _unbind_net_out(self, idx, input_seq):
        out = self.net(input_seq[:, idx:idx + self.wnd_size])

        if not isinstance(out, tuple):
            out = tuple([out])

        out_parts = [torch.unbind(x, dim=1) for x in out]

        return out_parts

    def _process_queues(self, queues, len_out):
        queues = [[None] * shift + queue + [None] * (len_out - len(queue) - shift)
                  for shift, queue in enumerate(queues)
                  ]

        """
        for queue in queues:
            print(list(isinstance(x, torch.Tensor) for x in queue))
        """

        for queue in queues:
            assert len_out == len(queue)

        out_frames = []

        for i in range(len_out):
            x = torch.stack([queue[i] for queue in queues if queue[i] is not None], dim=1)

            # print(x.shape[1])

            x = self.reduce(x)
            out_frames.append(x)

        out = torch.stack(out_frames, dim=1)
        return out

    def _forward_all(self, input_seq):
        len_input_seq = input_seq.shape[1]
        # len_out = len_input_seq - 2 * (self.wnd_size - 1)
        len_out = len_input_seq

        num_queues = self.wnd_size

        output_queues = None

        for queue_idx in range(num_queues):
            shift = queue_idx
            for queue_pos in range(shift, len_input_seq - (self.wnd_size - 1), self.wnd_size):
                out_parts = self._unbind_net_out(queue_pos, input_seq)

                if output_queues is None:
                    output_queues = [[[] for _ in range(num_queues)] for _ in range(len(out_parts))]

                for out_idx, out_part in enumerate(out_parts):
                    output_queues[out_idx][queue_idx].extend(out_part)

        """
        for x in output_queues:
            for y in x:
                print(len(y))
        """

        out = tuple(self._process_queues(queues, len_out) for queues in output_queues)

        if len(out) == 1:
            out = out[0]

        return out

    # B, F, C, H, W
    def forward(self, input_seq):
        if self.use_lazy_fwd:
            return self._forward_lazy(input_seq)
        else:
            return self._forward_all(input_seq)
