from torch import nn, Tensor

class CNNModel(nn.Module):
    def __init__(self, emb_dim : int = 32, out_dim : int = 100):
        super().__init__()
        self.emb_dim = emb_dim
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, emb_dim, (7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.MaxPool2d((3, 3), (2, 2), (1, 1))
        )
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(emb_dim, 2*emb_dim, (5, 5), padding=(2, 2), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*emb_dim, 4*emb_dim, (3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*emb_dim, 8*emb_dim, (1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*emb_dim, 8*emb_dim, (3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*emb_dim, 16*emb_dim, (1, 1), bias=False),
            nn.ReLU(inplace=True),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.output_layer = nn.Sequential(
            nn.Linear(16*emb_dim, 8*emb_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(8*emb_dim, out_dim, bias=False)
        )

    def forward(self, x : Tensor) -> Tensor:
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        flat_x = self.global_avg_pool(x).view(x.size(0), 16*self.emb_dim)
        res = self.output_layer(flat_x)
        return res
