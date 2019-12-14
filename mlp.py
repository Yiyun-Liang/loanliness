from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_1 = nn.Linear(651, 128)
        self.hidden_2 = nn.Linear(128, 64)
        self.hidden_3 = nn.Linear(64, 16)
        self.output = nn.Linear(16, 2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden_1(x)
        x = self.sigmoid(x)
        x = self.hidden_2(x)
        x = self.sigmoid(x)
        x = self.hidden_3(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x