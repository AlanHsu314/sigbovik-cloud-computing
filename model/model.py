import torch
from torch import nn

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.LeakyReLU, 
                 norm_layer=None, drop_prob=0.0):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = None if drop_prob == 0 else nn.Dropout(drop_prob)
        self.norm = None if norm_layer is None else norm_layer(out_dim)
        self.act = None if activation is None else activation()

    def forward(self, x):
        x = self.linear(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        inc = (out_dim - in_dim) / num_layers
        dims = [int(in_dim + i * inc) for i in range(num_layers)] + [out_dim]
        self.mlp = nn.Sequential(*[
            LinearLayer(dims[i], dims[i+1], 
                norm_layer=None if i == num_layers-1 else nn.BatchNorm1d,
                activation=None if i == num_layers-1 else nn.LeakyReLU)
            for i in range(num_layers)
        ])

class Autoencoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super(Autoencoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder = MLP(in_dim, hidden_dim, num_layers)
        self.decoder = MLP(hidden_dim, in_dim, num_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AutoencoderClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes, num_layers):
        super(AutoencoderClassifier, self).__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.classifier = MLP(hidden_dim, num_classes, num_layers)

class RNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super(RNNModel, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(in_dim, hidden_dim, )

# if __name__ == '__main__':
#     model = Autoencoder(768, 1, 3)
