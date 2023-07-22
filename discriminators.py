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
        nn.LeakyReLU(inplace=False),
        nn.Dropout1d(p=0.1, inplace=False),
    )


class PatchDiscriminator(nn.Module):
    def __init__(self, CHANNELS):
        super(PatchDiscriminator, self).__init__()
        
        self.conv1 = Conv_block(in_channels= CHANNELS, out_channels= 32, kernel_size = 3, stride = 2) # torch.Size([1, 32, 127])
        self.conv2 = Conv_block(in_channels= 32, out_channels= 64, kernel_size = 3, stride = 2)       # torch.Size([1, 64, 63])
        self.conv3 = Conv_block(in_channels= 64, out_channels= 128, kernel_size = 3, stride = 2)      # torch.Size([1, 128, 31])
        self.conv4 = Conv_block(in_channels= 128, out_channels= 256, kernel_size = 3, stride = 2)     # torch.Size([1, 256, 15])
        self.conv5 = Conv_block(in_channels= 256, out_channels= 1, kernel_size = 3, stride = 2)       # torch.Size([1, 1, 7])
        
        self.out = nn.Sigmoid()             # torch.Size([1, 1, 7])


    def forward(self, input):
        #print('input shape')
        #print(input.shape)      # torch.Size([1, 1, 256])
        x = self.conv1(input)
        #print(x.shape)          # torch.Size([1, 32, 127])
        x = self.conv2(x)
        #print(x.shape)          # torch.Size([1, 64, 63])
        x = self.conv3(x)
        #print(x.shape)          # torch.Size([1, 128, 31])
        x = self.conv4(x)
        #print(x.shape)          # torch.Size([1, 256, 15])
        x = self.conv5(x)
        #print(x.shape)          # torch.Size([1, 1, 7])
        x = self.out(x)
        #print(x.shape)          # torch.Size([1, 1, 7])
        return x
    

# def Conv_block(in_channels, out_channels, kernel_size , stride):
#     return nn.Sequential(
#         nn.Conv1d(in_channels, out_channels, kernel_size, stride),
#         nn.BatchNorm1d(out_channels),
#         nn.LeakyReLU(inplace=False),
#         nn.Dropout1d(p=0.1, inplace=False),
#     )

def Conv_block_no_norm(in_channels, out_channels, kernel_size , stride):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride),
        #nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(inplace=False),
        nn.Dropout1d(p=0.1, inplace=False),
    )

class SampleDiscriminator(nn.Module):
    def __init__(self, CHANNELS):
        super(SampleDiscriminator, self).__init__()

        
        self.conv1 = Conv_block(in_channels= CHANNELS, out_channels= 32, kernel_size = 4, stride = 3) 
        self.conv2 = Conv_block(in_channels= 32, out_channels= 64, kernel_size = 4, stride = 3)       
        self.conv3 = Conv_block(in_channels= 64, out_channels= 128, kernel_size = 4, stride = 3)      
        self.conv4 = Conv_block(in_channels= 128, out_channels= 256, kernel_size = 3, stride = 2)     
        self.conv5 = Conv_block_no_norm(in_channels= 256, out_channels= 1, kernel_size = 4, stride = 2) 
        
        self.out = nn.Sigmoid()             

    def forward(self, input):    
        # input: torch.Size([1, 1, 256]) 
        x = self.conv1(input)  # torch.Size([1, 32, 85])       
        x = self.conv2(x)      # torch.Size([1, 64, 28])     
        x = self.conv3(x)      # torch.Size([1, 128, 9])      
        x = self.conv4(x)      # torch.Size([1, 256, 4]) 
        x = self.conv5(x)      # torch.Size([1, 1, 1])     
        x = self.out(x)        # torch.Size([1, 1, 1])     
        return x


# class SampleDiscriminator(nn.Module):
#     def __init__(self, CHANNELS):
#         super(SampleDiscriminator, self).__init__()

        
#         self.conv1 = Conv_block(in_channels= CHANNELS, out_channels= 32, kernel_size = 4, stride = 3) 
#         self.conv2 = Conv_block(in_channels= 32, out_channels= 64, kernel_size = 4, stride = 3)       
#         self.conv3 = Conv_block(in_channels= 64, out_channels= 128, kernel_size = 4, stride = 3)      
#         self.conv4 = Conv_block(in_channels= 128, out_channels= 256, kernel_size = 3, stride = 2)     
#         self.conv5 = Conv_block_no_norm(in_channels= 256, out_channels= 1, kernel_size = 4, stride = 2) 
        
#         self.out = nn.Sigmoid()             


#     # input shape
#     # torch.Size([1, 1, 256])
#     # torch.Size([1, 32, 85])
#     # torch.Size([1, 64, 28])
#     # torch.Size([1, 128, 9])
#     # torch.Size([1, 256, 4])
#     # torch.Size([1, 1, 1])
#     # torch.Size([1, 1, 1])

#     def forward(self, input):    
#         # input: torch.Size([1, 1, 256]) 
#         x = self.conv1(input)  # torch.Size([1, 32, 85])       
#         x = self.conv2(x)      # torch.Size([1, 64, 28])     
#         x = self.conv3(x)      # torch.Size([1, 128, 9])      
#         x = self.conv4(x)      # torch.Size([1, 256, 4]) 
#         x = self.conv5(x)      # torch.Size([1, 1, 1])     
#         x = self.out(x)        # torch.Size([1, 1, 1])     
#         return x