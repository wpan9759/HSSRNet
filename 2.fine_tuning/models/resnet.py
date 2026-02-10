import torch.nn as nn

class Dense(nn.Module):
    def __init__(self, drop, in_size, out_size):
        super(Dense, self).__init__()
        
        self.linear = nn.Linear(in_size, out_size)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.linear(x)
        x = self.prelu(x)
        x = self.dropout(x)
        
        return x


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.prelu = nn.PReLU()  # Apply after the skip connection

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)  # Activation after batch norm

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Merge skip connection and output, then apply activation
        out += identity
        out = self.prelu(out)

        return out


class ResNet1D(nn.Module):
    def __init__(self, num_classes, input_length):
        super(ResNet1D, self).__init__()
        
        self.input_length = input_length
        
        self.layer0 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            )

        # Residual blocks
        self.layer1 = self.make_layer(16, 32, kernel_size=3, stride=2, num_blocks=3)
        self.layer2 = self.make_layer(32, 64, kernel_size=3, stride=2, num_blocks=3)

        self.flatten = nn.Flatten(start_dim=1)

        # Calculate the output size after conv and pooling layers
        conv_output_size = self._get_conv_output(input_length)

        # Dense layers
        self.dense1 = Dense(0.25, conv_output_size, 128)
        self.dense2 = Dense(0.1, 128, 64)
        self.linear = nn.Linear(64, num_classes)

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

    def _get_conv_output(self, input_length):
        def conv1d_output_length(l_in, kernel_size, stride, padding):
            return (l_in + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        length = input_length
        length = conv1d_output_length(length, 7, 2, 3)  # Conv1
        length = conv1d_output_length(length, 3, 2, 1)  # MaxPool
        length = conv1d_output_length(length, 3, 2, 1)  # Layer1
        length = conv1d_output_length(length, 3, 2, 1)  # Layer2

        return 64 * length

    def forward(self, x):
        x = self.layer0(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.linear(x)

        return x
    
