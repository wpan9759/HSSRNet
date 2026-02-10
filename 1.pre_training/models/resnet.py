import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.prelu = nn.PReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)


        out += identity
        out = self.prelu(out)

        return out

class ResNet1D(nn.Module):
    def __init__(self):
        super(ResNet1D, self).__init__()


        self.layer0 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            )


        self.layer1 = self.make_layer(16, 32, kernel_size=3, stride=2, num_blocks=3)
        self.layer2 = self.make_layer(32, 64, kernel_size=3, stride=2, num_blocks=3)

        self.flatten = nn.Flatten(start_dim=1)


        self.dense = nn.Sequential(
            nn.Linear(8640, 4096),
            nn.PReLU(),
            nn.Linear(4096, 256)
        )

    def make_layer(self, in_channels, out_channels, kernel_size, stride, num_blocks):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, kernel_size, stride, downsample))

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.flatten(x)
        
        x = self.dense(x)
        
        return x