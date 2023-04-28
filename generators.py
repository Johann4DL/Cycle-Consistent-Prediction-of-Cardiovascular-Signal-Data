import torch
import torch.nn as nn

def double_conv_pad(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='zeros'),
        nn.LeakyReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='zeros'),
        nn.LeakyReLU(inplace=True)
    )

# This one has skip connections
class OneChannelUnetGenerator(nn.Module):
    def __init__(self):
        super(OneChannelUnetGenerator, self).__init__()
        self.maxpool = nn.MaxPool1d(2)

        self.down_conv1 = double_conv_pad(1, 32) 
        self.down_conv2 = double_conv_pad(32, 64) 
        self.down_conv3 = double_conv_pad(64, 128)
        self.down_conv4 = double_conv_pad(128, 256)

        self.up_trans1 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.up_conv1 = double_conv_pad(256, 128)
        self.up_trans2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.up_conv2 = double_conv_pad(128, 64)
        self.up_trans3 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.up_conv3 = double_conv_pad(64, 32)

        self.out = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, input):
        
        # batch_size, channels, tensor_size
        # downsampling
        x1 = self.down_conv1(input)     
        x2 = self.maxpool(x1) 
        x3 = self.down_conv2(x2)  
        x4 = self.maxpool(x3) 
        x5 = self.down_conv3(x4)   
        x6 = self.maxpool(x5) 
        x7 = self.down_conv4(x6)

        # upsampling
        x = self.up_trans1(x7)
        x = self.up_conv1(torch.cat([x, x5], 1))
        x = self.up_trans2(x)
        x = self.up_conv2(torch.cat([x, x3], 1))
        x = self.up_trans3(x)
        x = self.up_conv3(torch.cat([x, x1], 1))
        x = self.out(x)
        return x
    

class MultiChannelUnetGenerator(nn.Module):
    def __init__(self, INPUTCHANNELS, OUTPUTCHANNELS):
        super(MultiChannelUnetGenerator, self).__init__()
        self.maxpool = nn.MaxPool1d((2))  


        self.down_conv1 = double_conv_pad(INPUTCHANNELS, 32) 
        self.down_conv2 = double_conv_pad(32, 64) 
        self.down_conv3 = double_conv_pad(64, 128)
        self.down_conv4 = double_conv_pad(128, 256)

        self.up_trans1 = nn.ConvTranspose1d(256, 128, kernel_size=(2), stride=2, padding=0)
        self.up_conv1 = double_conv_pad(128, 128)
        self.up_trans2 = nn.ConvTranspose1d(128, 64, kernel_size=(2), stride=2, padding=0)
        self.up_conv2 = double_conv_pad(64, 64)
        self.up_trans3 = nn.ConvTranspose1d(64, 32, kernel_size=(2), stride=2, padding=0)
        self.up_conv3 = double_conv_pad(32, 32)

        self.out = nn.Conv1d(32, OUTPUTCHANNELS, kernel_size=1) # kernel_size must be == 1

    def forward(self, input):
        # [Batch size, Channels in, Height, Width]
        x1 = self.down_conv1(input) 
        x2 = self.maxpool(x1) 
        x3 = self.down_conv2(x2)
        x4 = self.maxpool(x3) 
        x5 = self.down_conv3(x4) 
        x6 = self.maxpool(x5)  
        x7 = self.down_conv4(x6)

        # # decoder
        x = self.up_trans1(x7)
        x = self.up_conv1(x)
        x = self.up_trans2(x)
        x = self.up_conv2(x)
        x = self.up_trans3(x)
        x = self.up_conv3(x)
        x = self.out(x)

        return x
    
class SkipConnectionsMultiChannelUnetGenerator(nn.Module):
    def __init__(self, INPUTCHANNELS, OUTPUTCHANNELS):
        super(SkipConnectionsMultiChannelUnetGenerator, self).__init__()
        self.maxpool = nn.MaxPool1d((2))  

        self.down_conv1 = double_conv_pad(INPUTCHANNELS, 32) 
        self.down_conv2 = double_conv_pad(32, 64) 
        self.down_conv3 = double_conv_pad(64, 128)
        self.down_conv4 = double_conv_pad(128, 256)

        self.up_trans1 = nn.ConvTranspose1d(256, 128, kernel_size=(2), stride=2, padding=0)
        self.up_conv1 = double_conv_pad(256, 128)
        self.up_trans2 = nn.ConvTranspose1d(128, 64, kernel_size=(2), stride=2, padding=0)
        self.up_conv2 = double_conv_pad(128, 64)
        self.up_trans3 = nn.ConvTranspose1d(64, 32, kernel_size=(2), stride=2, padding=0)
        self.up_conv3 = double_conv_pad(64, 32)

        self.out = nn.Conv1d(32, OUTPUTCHANNELS, kernel_size=1) # kernel_size must be == 1

    def forward(self, input):
        # [Batch size, Channels in, Height, Width]
        
        # downsampling
        x1 = self.down_conv1(input)   
        x2 = self.maxpool(x1) 
        x3 = self.down_conv2(x2)  
        x4 = self.maxpool(x3) 
        x5 = self.down_conv3(x4)  
        x6 = self.maxpool(x5) 
        x7 = self.down_conv4(x6)

        # upsampling
        x = self.up_trans1(x7)
        x = self.up_conv1(torch.cat([x, x5], 1))  # skip connection
        x = self.up_trans2(x)
        x = self.up_conv2(torch.cat([x, x3], 1))  # skip connection
        x = self.up_trans3(x)
        x = self.up_conv3(torch.cat([x, x1], 1))  # skip connection
        x = self.out(x)
        return x
    
class EmbeddingUnetGenerator(nn.Module):
    def __init__(self, INPUTCHANNELS, OUTPUTCHANNELS):
        super(EmbeddingUnetGenerator, self).__init__()
        self.maxpool = nn.MaxPool1d((2))  

        self.down_conv1 = double_conv_pad(INPUTCHANNELS, 32) 
        self.down_conv2 = double_conv_pad(32, 64) 
        self.down_conv3 = double_conv_pad(64, 128)
        self.down_conv4 = double_conv_pad(128, 256)

        self.embedding = torch.nn.Embedding(num_embeddings=5, embedding_dim=32)

        self.up_trans1 = nn.ConvTranspose1d(256, 128, kernel_size=(2), stride=2, padding=0)
        self.up_conv1 = double_conv_pad(256, 128)
        self.up_trans2 = nn.ConvTranspose1d(128, 64, kernel_size=(2), stride=2, padding=0)
        self.up_conv2 = double_conv_pad(128, 64)
        self.up_trans3 = nn.ConvTranspose1d(64, 32, kernel_size=(2), stride=2, padding=0)
        self.up_conv3 = double_conv_pad(64, 32)

        self.out = nn.Conv1d(32, OUTPUTCHANNELS, kernel_size=1) # kernel_size must be == 1

    def forward(self, input, phase):
        # [Batch size, Channels in, Height, Width]
        
        # downsampling
        x1 = self.down_conv1(input)   
        x2 = self.maxpool(x1) 
        x3 = self.down_conv2(x2)  
        x4 = self.maxpool(x3) 
        x5 = self.down_conv3(x4)  
        x6 = self.maxpool(x5) 
        x7 = self.down_conv4(x6)

        # upsampling
        e = self.embedding(phase)
        x7 = x7 + e
        x = self.up_trans1(x7)
        x = self.up_conv1(torch.cat([x, x5], 1))  # skip connection
        x = self.up_trans2(x)
        x = self.up_conv2(torch.cat([x, x3], 1))  # skip connection
        x = self.up_trans3(x)
        x = self.up_conv3(torch.cat([x, x1], 1))  # skip connection
        x = self.out(x)
        return x