import torch
import torch.nn as nn

"""
GAN의 Generator를 자유롭게 개선해주세요!!
단순 논문 구현한 Generator에 이것 저것을 추가해도 좋고, 변경해도 좋습니다!

Hint:
1. Batch Normalization
2. Dropout
3. Deep Layer
4. etc...

Layer가 깊을수록 성능이 좋아질까요??
 
"""

class Generator(nn.Module):

    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256), 
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, img_dim), 
            nn.Tanh() 
        )

    
    def forward(self, z):
        #One Line
        return self.gen(z)
