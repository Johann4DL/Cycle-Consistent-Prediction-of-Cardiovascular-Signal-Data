import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.InstanceNorm1d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.block(x)
    

class OneChannelDiscriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=[32,64,128,256]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, out_channels[0], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        layers = []
        in_channels = out_channels[0]
        for out_channel in out_channels[1:]:
            layers.append(Block(in_channels, out_channel, stride=1 if out_channel == 256 else 2))
            in_channels = out_channel

        layers.append(nn.Conv1d(out_channels[-1], 1, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))


def Conv_block(in_channels, out_channels, kernel_size , stride):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Dropout1d(p=0.1, inplace=False),
    )


class MultiChannelDiscriminator(nn.Module):
    def __init__(self, CHANNELS):
        super(MultiChannelDiscriminator, self).__init__()
        
        self.conv1 = Conv_block(in_channels= CHANNELS, out_channels= 32, kernel_size = 3, stride = 2) # remove padding
        self.conv2 = Conv_block(in_channels= 32, out_channels= 64, kernel_size = 3, stride = 2)
        self.conv3 = Conv_block(in_channels= 64, out_channels= 128, kernel_size = 3, stride = 2)
        self.conv4 = Conv_block(in_channels= 128, out_channels= 256, kernel_size = 3, stride = 2)
        self.conv5 = Conv_block(in_channels= 256, out_channels= 1, kernel_size = 3, stride = 2)
        
        self.out = nn.Sigmoid()
        # self.out = nn.BCEWithLogitsLoss()


    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.out(x)
        return x