class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_units=128, output_dim=10):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim)
        )

    def forward(self, x):
        return self.network(x)