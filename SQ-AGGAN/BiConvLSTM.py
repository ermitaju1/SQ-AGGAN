#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

class BiConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(BiConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        # NOTE: This keeps height and width the same
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        # # TODO: we may want this to be different than the conv we use inside each cell
        # self.conv_concat = nn.Conv2d(in_channels=2 * self.hidden_dim, #in_channels=self.input_dim + self.hidden_dim,
        #                              out_channels=self.hidden_dim,
        #                              kernel_size=self.kernel_size,
        #                              padding=self.padding,
        #                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        #h_cur은 input이 들어가는 line, c_cur은 cell state
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        # print("input_tensor.shape", input_tensor.shape)   # (b, 256, 8, 8)
        # print("h_cur.shape", h_cur.shape)                 # (b, 512, 8, 8)
        # print("combined.shape", combined.shape)           # (b, 768, 8, 8)

        combined_conv = self.conv(combined) ##결합한거에 conv를 적용해
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        
        o = torch.sigmoid(cc_o)
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class BiConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, device,
                 bias=True, return_all_layers=False):
        super(BiConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers) #input: (param, num_layer)
        
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        ## kernel size, hidden dim, num layer랑 다 같은 크기여야해서
        
        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.device = device
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(BiConvLSTMCell(input_size=(self.height, self.width),
                                            input_dim=cur_input_dim,
                                            hidden_dim=self.hidden_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor):
        hidden_state = self._init_hidden(batch_size=input_tensor.size(0), device=self.device)

        layer_output_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            backward_states = []
            forward_states = []
            output_inner = []

            #######이해가 안 됨
            #hidden backward, cur backward
            hb, cb = hidden_state[layer_idx]
            for t in range(seq_len):
                hb, cb = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, seq_len - t - 1, :, :, :], cur_state=[hb, cb])
                backward_states.append(hb)

            hf, cf = hidden_state[layer_idx]
            for t in range(seq_len):
                hf, cf = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[hf, cf])
                forward_states.append(hf)
           
        #    ## conv & concat version.....
        #     for t in range(seq_len):
        #         h = self.cell_list[layer_idx].conv_concat(torch.cat((forward_states[t], backward_states[seq_len - t - 1]), dim=1))
        #         output_inner.append(h)

            for t in range(seq_len):
                h = forward_states[t] + backward_states[seq_len - t - 1]
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)   # 3 * (b, 512, 8, 8) -> (b, 3, 512, 8, 8)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)            # (num_layers, b, 3, 512, 8, 8)

        if not self.return_all_layers:                        # becasue "return_all_layers=False"!!!!!  -> (1, b, 3, 512, 8, 8)
            return layer_output_list[-1]

        return layer_output_list

    def _init_hidden(self, batch_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append((Variable(torch.zeros(batch_size, self.hidden_dim[i], self.height, self.width)).to(device),
                                Variable(torch.zeros(batch_size, self.hidden_dim[i], self.height, self.width)).to(device)))
        return init_states
        ##초기tensor 값을 0로 초기화 하기 위함 
    
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):  ##list인지 아닌지 판단
            param = [param] * num_layers
        return param
