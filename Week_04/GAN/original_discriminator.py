import torch
import torch.nn as nn

"""
정말 간단하게 논문 구현만 해봅시다
논문에서 제시한대로 구현하면 됩니다!!
데이터셋은 Fashion MNIST를 사용하기 때문에, 이미지의 크기(=in_features)는 28 * 28 = 784입니다

Hint: nn.Sequential을 사용하면 간단하게 구현할 수 있습니다.

"""

class Maxout(nn.Module):
    def __init__(self, in_features, out_features, num_pieces):
        super(Maxout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_pieces = num_pieces
        self.linear = nn.Linear(in_features, out_features * num_pieces)

    def forward(self, x):
        x = self.linear(x)
        x, _ = x.view(-1, self.out_features, self.num_pieces).max(dim=2)
        return x

class Original_Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.maxout_layer = Maxout(in_features, 128, num_pieces=5)
        self.disc = nn.Sequential(  
            nn.Linear(128, 1),
            nn.Dropout(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(self.maxout_layer(x))