import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def calculate_Lout_conv(Lin=20,kernel=1, padding=0, dilation=1, stride=1):
    return (Lin + 2*padding - dilation*(kernel - 1)- 1)/stride + 1


class RNNAutoencoder(nn.Module):
    def __init__(self, n_inputs=8, hidden_size=20, bidirectional=False, n_layers=1, seqlen=60, batch_size=64):
        super(RNNAutoencoder, self).__init__()
        self.n_inputs = n_inputs
        n_outputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.seqlen = seqlen
        self.teacher_forcing_prob = 1
        self.period_decay = 50
        if bidirectional:
            self.directions = 2
        else:
            self.directions = 1
        self.enc_gru = nn.GRU(input_size=n_inputs, num_layers=n_layers, hidden_size=hidden_size,
                              batch_first=True, bidirectional=bidirectional)
        self.dec_gru = nn.GRU(input_size=n_inputs, num_layers=n_layers, hidden_size=hidden_size*self.directions, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size*self.directions, out_features=n_outputs)

        for name, param in self.named_parameters():
            if 'bias' not in name:
                torch.nn.init.xavier_normal_(param.data)

    def decay_teacher(self, it):
        self.teacher_forcing_prob = max(0., 1 - it*(0.8/self.period_decay))


    def forward(self, x):
        # Encoder
        size = x.shape
        h = self.init_hidden(size[0])
        _, h = self.enc_gru(x, h)

        # Decoder
        outs = []
        use_teacher_forcing = True if random.random() < self.teacher_forcing_prob else False
        if not self.training:
            use_teacher_forcing = False

        if use_teacher_forcing:
            x_out, h_out = self.dec_gru(x, h.reshape(self.n_layers,  size[0], -1))
            x_out = self.linear(x_out.reshape(size[0], size[1], self.hidden_size*self.directions))
            outs.append(x_out)

        else:
            x_out = x[:, 0, :].reshape(size[0], 1, self.n_inputs)
            h_out = h.reshape(self.n_layers, size[0], -1)
            for i in range(size[1]):
                x_out, h_out = self.dec_gru(x_out, h_out)
                x_out = x_out.reshape(size[0], 1, self.hidden_size * self.directions)  # La primera dimensión debería ser size[0]*size[1] pero en este caso size[1] es 1
                x_out = self.linear(x_out).reshape(size[0], 1, self.n_inputs)
                outs.append(x_out)

        outs = torch.cat(outs, dim=1)

        return outs, h


    def encoder(self, x):# Encoder
        size = x.shape
        h = self.init_hidden(size[0])
        _, h = self.enc_gru(x, h)
        return x, h

    def decoder(self, x, h):
        # Decoder
        size = x.shape
        outs = []
        use_teacher_forcing = True if random.random() < self.teacher_forcing_prob else False
        if not self.training:
            use_teacher_forcing = False

        if use_teacher_forcing:
            x_out, h_out = self.dec_gru(x, h.reshape(self.n_layers, size[0], -1))
            x_out = self.linear(x_out.reshape(size[0], size[1], self.hidden_size * self.directions))
            outs.append(x_out)

        else:
            x_out = x[:, 0, :].reshape(size[0], 1, self.n_inputs)
            h_out = h.reshape(self.n_layers, size[0], -1)
            for i in range(size[1]):
                x_out, h_out = self.dec_gru(x_out, h_out)
                x_out = x_out.reshape(size[0], 1,
                                      self.hidden_size * self.directions)  # La primera dimensión debería ser size[0]*size[1] pero en este caso size[1] es 1
                x_out = self.linear(x_out).reshape(size[0], 1, self.n_inputs)
                outs.append(x_out)

        outs = torch.cat(outs, dim=1)

        return outs, h

    def init_hidden(self, batch_size):
        return torch.randn(self.n_layers * self.directions, batch_size, self.hidden_size).float().to(
            self.enc_gru.all_weights[0][0].device)


class RNNAutoencoder_AddLayer(nn.Module):
    def __init__(self, n_inputs=8, hidden_size=20, bidirectional=False, n_layers=1, seqlen=60, batch_size=64):
        super(RNNAutoencoder_AddLayer, self).__init__()
        self.n_inputs = n_inputs
        n_outputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.seqlen = seqlen
        self.teacher_forcing_prob = 1
        self.period_decay = 50
        if bidirectional:
            self.directions = 2
        else:
            self.directions = 1
        self.enc_gru = nn.GRU(input_size=n_inputs, num_layers=n_layers, hidden_size=hidden_size,
                              batch_first=True, bidirectional=bidirectional)
        self.dec_gru = nn.GRU(input_size=n_inputs, num_layers=n_layers, hidden_size=hidden_size*self.directions, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size*self.directions, out_features=n_outputs)
        self.linear_nce1 = nn.Linear(hidden_size, int(hidden_size/4))
        self.linear_nce2 = nn.Linear(int(hidden_size/4), int(hidden_size/4))

        for name, param in self.named_parameters():
            if 'bias' not in name:
                torch.nn.init.xavier_normal_(param.data)

    def decay_teacher(self, it):
        self.teacher_forcing_prob = max(0., 1 - it*(0.8/self.period_decay))


    def forward(self, x):
        # Encoder
        size = x.shape
        h = self.init_hidden(size[0])
        _, h = self.enc_gru(x, h)
        h_nce = self.linear_nce2(torch.relu(self.linear_nce1(h[-1, :, :])))

        # Decoder
        outs = []
        use_teacher_forcing = True if random.random() < self.teacher_forcing_prob else False
        if not self.training:
            use_teacher_forcing = False

        if use_teacher_forcing:
            x_out, h_out = self.dec_gru(x, h.reshape(self.n_layers,  size[0], -1))
            x_out = self.linear(x_out.reshape(size[0], size[1], self.hidden_size*self.directions))
            outs.append(x_out)

        else:
            x_out = x[:, 0, :].reshape(size[0], 1, self.n_inputs)
            h_out = h.reshape(self.n_layers, size[0], -1)
            for i in range(size[1]):
                x_out, h_out = self.dec_gru(x_out, h_out)
                x_out = x_out.reshape(size[0], 1, self.hidden_size * self.directions)  # La primera dimensión debería ser size[0]*size[1] pero en este caso size[1] es 1
                x_out = self.linear(x_out).reshape(size[0], 1, self.n_inputs)
                outs.append(x_out)

        outs = torch.cat(outs, dim=1)

        return outs, h_nce, h


    def encoder(self, x):# Encoder
        size = x.shape
        h = self.init_hidden(size[0])
        _, h = self.enc_gru(x, h)
        h_nce = self.linear_nce2(torch.tanh(self.linear_nce1(h[-1, :, :])))
        return x, h_nce

    def decoder(self, x, h):
        # Decoder
        size = x.shape
        outs = []
        use_teacher_forcing = True if random.random() < self.teacher_forcing_prob else False
        if not self.training:
            use_teacher_forcing = False

        if use_teacher_forcing:
            x_out, h_out = self.dec_gru(x, h.reshape(self.n_layers, size[0], -1))
            x_out = self.linear(x_out.reshape(size[0], size[1], self.hidden_size * self.directions))
            outs.append(x_out)

        else:
            x_out = x[:, 0, :].reshape(size[0], 1, self.n_inputs)
            h_out = h.reshape(self.n_layers, size[0], -1)
            for i in range(size[1]):
                x_out, h_out = self.dec_gru(x_out, h_out)
                x_out = x_out.reshape(size[0], 1,
                                      self.hidden_size * self.directions)  # La primera dimensión debería ser size[0]*size[1] pero en este caso size[1] es 1
                x_out = self.linear(x_out).reshape(size[0], 1, self.n_inputs)
                outs.append(x_out)

        outs = torch.cat(outs, dim=1)

        return outs, h

    def init_hidden(self, batch_size):
        return torch.randn(self.n_layers * self.directions, batch_size, self.hidden_size).float().to(
            self.enc_gru.all_weights[0][0].device)



