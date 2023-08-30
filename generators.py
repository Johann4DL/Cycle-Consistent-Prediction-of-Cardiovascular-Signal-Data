import torch
import torch.nn as nn
import torch.nn.functional as F
import generators

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def conv_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
    )

def single_conv_pad(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='zeros'),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Dropout1d(p=0.1, inplace=False),
    )

# Try out InstanceNorm1d instead of BatchNorm1d
def double_conv_pad(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='zeros'),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Dropout1d(p=0.1, inplace=False),
        nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='zeros'),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Dropout1d(p=0.1, inplace=False),
    )

def triple_conv_pad(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='zeros'),
        nn.LeakyReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='zeros'),
        nn.LeakyReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='zeros'),
        nn.LeakyReLU(inplace=True)
    )

    

class BasicGenerator(nn.Module):
    def __init__(self, INPUTCHANNELS, OUTPUTCHANNELS):
        super(BasicGenerator, self).__init__()
        '''
        A Basic Unet Generator without skip connections
        '''

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


def residual_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride = 1, padding=0),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, kernel_size = 1, stride = 1, padding=0),
        nn.BatchNorm1d(out_channels),
    )


    
class SkipConGenerator(nn.Module):
    def __init__(self, INPUTCHANNELS, OUTPUTCHANNELS):
        super(SkipConGenerator, self).__init__()
        '''
        A Basic Unet Generator with skip connections
        '''
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



class SkipTensorEmbeddingGen(nn.Module):
    def __init__(self, INPUTCHANNELS, OUTPUTCHANNELS, Down = True, Bottleneck = True):
        super(SkipTensorEmbeddingGen, self).__init__()
        self.Down = Down
        self.Bottleneck = Bottleneck
        self.maxpool = nn.MaxPool1d((2))  

        self.source_intervention = torch.nn.Embedding(num_embeddings=11, embedding_dim=256)
        self.source_phase = torch.nn.Embedding(num_embeddings=6, embedding_dim=256)

        self.down_conv1 = double_conv_pad(INPUTCHANNELS, 32) 
        self.down_conv2 = double_conv_pad(32, 64) 
        self.down_conv3 = double_conv_pad(64, 128)
        self.down_conv4 = double_conv_pad(128, 256)

        self.target_intervention = torch.nn.Embedding(num_embeddings=11, embedding_dim=32)
        self.target_phase = torch.nn.Embedding(num_embeddings=6, embedding_dim=32)

        self.up_trans1 = nn.ConvTranspose1d(256, 128, kernel_size=(2), stride=2, padding=0)
        self.up_conv1 = double_conv_pad(256, 128)
        self.up_trans2 = nn.ConvTranspose1d(128, 64, kernel_size=(2), stride=2, padding=0)
        self.up_conv2 = double_conv_pad(128, 64)
        self.up_trans3 = nn.ConvTranspose1d(64, 32, kernel_size=(2), stride=2, padding=0)
        self.up_conv3 = double_conv_pad(64, 32)

        self.out = nn.Conv1d(32, OUTPUTCHANNELS, kernel_size=1) # kernel_size must be == 1


    def forward(self, input, source_phase, source_intervention, target_phase, target_intervention):
        if self.Down:
            sp = self.source_phase(source_phase)
            si = self.source_intervention(source_intervention)
            input += sp + si 
        x1 = self.down_conv1(input) 
        x2 = self.maxpool(x1) 
        x3 = self.down_conv2(x2)
        x4 = self.maxpool(x3) 
        x5 = self.down_conv3(x4) 
        x6 = self.maxpool(x5)  
        x7 = self.down_conv4(x6)

        # # decoder
        if self.Bottleneck:
            tp = self.target_phase(target_phase)  
            ti = self.target_intervention(target_intervention)
            x7 += tp + ti # target embeddings are added before upsampling  # lieber concatenating und dann fully connected to amtch the dimension
        x = self.up_trans1(x7)
        x = self.up_conv1(torch.cat([x, x5], 1))
        x = self.up_trans2(x)
        x = self.up_conv2(torch.cat([x, x3], 1))
        x = self.up_trans3(x)
        x = self.up_conv3(torch.cat([x, x1], 1))
        x = self.out(x)

        return x



class TensorEmbeddingGen(nn.Module):
    def __init__(self, INPUTCHANNELS, OUTPUTCHANNELS, Down = True, Bottleneck = True):
        super(TensorEmbeddingGen, self).__init__()
        self.Down = Down
        self.Bottleneck = Bottleneck
        self.maxpool = nn.MaxPool1d((2))  

        self.source_intervention = torch.nn.Embedding(num_embeddings=11, embedding_dim=256)
        self.source_phase = torch.nn.Embedding(num_embeddings=7, embedding_dim=256)

        self.down_conv1 = double_conv_pad(INPUTCHANNELS, 32) 
        self.down_conv2 = double_conv_pad(32, 64) 
        self.down_conv3 = double_conv_pad(64, 128)
        self.down_conv4 = double_conv_pad(128, 256)

        self.target_intervention = torch.nn.Embedding(num_embeddings=11, embedding_dim=32)
        self.target_phase = torch.nn.Embedding(num_embeddings=7, embedding_dim=32)

        self.up_trans1 = nn.ConvTranspose1d(256, 128, kernel_size=(2), stride=2, padding=0)
        self.up_conv1 = double_conv_pad(128, 128)
        self.up_trans2 = nn.ConvTranspose1d(128, 64, kernel_size=(2), stride=2, padding=0)
        self.up_conv2 = double_conv_pad(64, 64)
        self.up_trans3 = nn.ConvTranspose1d(64, 32, kernel_size=(2), stride=2, padding=0)
        self.up_conv3 = double_conv_pad(32, 32)

        self.out = nn.Conv1d(32, OUTPUTCHANNELS, kernel_size=1) # kernel_size must be == 1


    def forward(self, input, source_phase, source_intervention, target_phase, target_intervention):
        if self.Down:
            sp = self.source_phase(source_phase)
            si = self.source_intervention(source_intervention)
            input += sp + si 
        x1 = self.down_conv1(input) 
        x2 = self.maxpool(x1) 
        x3 = self.down_conv2(x2)
        x4 = self.maxpool(x3) 
        x5 = self.down_conv3(x4) 
        x6 = self.maxpool(x5)  
        x7 = self.down_conv4(x6)

        # # decoder
        if self.Bottleneck:
            tp = self.target_phase(target_phase)  
            ti = self.target_intervention(target_intervention)
            x7 += tp + ti # target embeddings are added before upsampling  # lieber concatenating und dann fully connected to amtch the dimension
        x = self.up_trans1(x7)
        x = self.up_conv1(x)
        x = self.up_trans2(x)
        x = self.up_conv2(x)
        x = self.up_trans3(x)
        x = self.up_conv3(x)
        x = self.out(x)

        return x



class SkipOneHotGenerator(nn.Module):
    def __init__(self, INPUTCHANNELS, OUTPUTCHANNELS, WINDOWSIZE, Down = True, Bottleneck = True):
        super(SkipOneHotGenerator, self).__init__()
        self.WINDOWSIZE = WINDOWSIZE
        self.Down = Down
        self.Bottleneck = Bottleneck
        self.maxpool = nn.MaxPool1d((2))  

        self.sourcephaseLinear = nn.Linear(7, 1)
        self.sourceinterventionLinear = nn.Linear(12, 1)
        self.sourceFCLinear = nn.Linear(768, 256)

        self.down_conv1 = double_conv_pad(INPUTCHANNELS, 32) 
        self.down_conv2 = double_conv_pad(32, 64) 
        self.down_conv3 = double_conv_pad(64, 128)
        self.down_conv4 = double_conv_pad(128, 256)

        self.targetphaseLinear = nn.Linear(7, 32)
        self.targetinterventionLinear = nn.Linear(12, 32)
        self.targetFCLinear = nn.Linear(96, 32)

        self.up_trans1 = nn.ConvTranspose1d(256, 128, kernel_size=(2), stride=2, padding=0)
        self.up_conv1 = double_conv_pad(256, 128)
        self.up_trans2 = nn.ConvTranspose1d(128, 64, kernel_size=(2), stride=2, padding=0)
        self.up_conv2 = double_conv_pad(128, 64)
        self.up_trans3 = nn.ConvTranspose1d(64, 32, kernel_size=(2), stride=2, padding=0)
        self.up_conv3 = double_conv_pad(64, 32)

        self.out = nn.Conv1d(32, OUTPUTCHANNELS, kernel_size=1) # kernel_size must be == 1


    def forward(self, input, source_phase, source_intervention, target_phase, target_intervention):
        # downsampling
        if self.Down:
            pS = F.one_hot(source_phase, num_classes=7).type(torch.FloatTensor).to(DEVICE)  #torch.Size([1, 256, 6])
            iS = F.one_hot(source_intervention, num_classes=12).type(torch.FloatTensor).to(DEVICE) #torch.Size([1, 256, 11])
            pS = self.sourcephaseLinear(pS).to(DEVICE)
            iS = self.sourceinterventionLinear(iS).to(DEVICE)
            pS = pS.reshape(input.shape[0], input.shape[1], input.shape[2]).to(DEVICE)
            iS = iS.reshape(input.shape[0], input.shape[1], input.shape[2]).to(DEVICE)
            # Concatenate the phase and intervention one hot encodings with the output of the last convolutional layer and reshape
            input = torch.cat([input, pS, iS], 2) # torch.Size([1, 1, 768])\
            input = self.sourceFCLinear(input).to(DEVICE) 

        x1 = self.down_conv1(input)   
        x2 = self.maxpool(x1) 
        x3 = self.down_conv2(x2)  
        x4 = self.maxpool(x3) 
        x5 = self.down_conv3(x4) 
        x6 = self.maxpool(x5) 
        x7 = self.down_conv4(x6)     

        # upsampling
        if self.Bottleneck:
            # one hot encoding
            pT = F.one_hot(target_phase, num_classes=7).type(torch.FloatTensor).to(DEVICE)
            iT = F.one_hot(target_intervention, num_classes=12).type(torch.FloatTensor).to(DEVICE)
            # Fully connected Layers
            pT = self.targetphaseLinear(pT).to(DEVICE)
            iT = self.targetinterventionLinear(iT).to(DEVICE)
            # Concatenate the phase and intervention one hot encodings with the output of the last convolutional layer and reshape
            x7 = torch.cat([x7, pT, iT], 1) 
            x7 = x7.reshape(x7.shape[0], self.WINDOWSIZE, 96)
            x7 = self.targetFCLinear(x7).to(DEVICE) 
        
        x = self.up_trans1(x7)
        x = self.up_conv1(torch.cat([x, x5], 1))  # skip connection
        x = self.up_trans2(x)
        x = self.up_conv2(torch.cat([x, x3], 1))  # skip connection
        x = self.up_trans3(x)
        x = self.up_conv3(torch.cat([x, x1], 1))  # skip connection
        x = self.out(x)
        return x



class OneHotGenerator(nn.Module):
    def __init__(self, INPUTCHANNELS, OUTPUTCHANNELS, WINDOWSIZE, Down = False, Bottleneck = True):
        super(OneHotGenerator, self).__init__()
        self.WINDOWSIZE = WINDOWSIZE
        self.Down = Down
        self.Bottleneck = Bottleneck
        self.maxpool = nn.MaxPool1d((2))  

        self.ReshapeDown = conv_layer(256, 1)
        self.DownFCLayer = nn.Linear(271, 256)

        self.down_conv1 = double_conv_pad(INPUTCHANNELS, 32) 
        self.down_conv2 = double_conv_pad(32, 64) 
        self.down_conv3 = double_conv_pad(64, 128)
        self.down_conv4 = double_conv_pad(128, 256)

        self.BottleNeckFCLinear = nn.Linear(47, 32)

        self.up_trans1 = nn.ConvTranspose1d(256, 128, kernel_size=(2), stride=2, padding=0)
        self.up_conv1 = double_conv_pad(128, 128)
        self.up_trans2 = nn.ConvTranspose1d(128, 64, kernel_size=(2), stride=2, padding=0)
        self.up_conv2 = double_conv_pad(64, 64)
        self.up_trans3 = nn.ConvTranspose1d(64, 32, kernel_size=(2), stride=2, padding=0)
        self.up_conv3 = double_conv_pad(32, 32)

        self.out = nn.Conv1d(32, OUTPUTCHANNELS, kernel_size=1) # kernel_size must be == 1


    def forward(self, input, source_phase, source_intervention, target_phase, target_intervention):
        # downsampling
        if self.Down:  # input shape:  torch.Size([1, 1, 256]) 
            pS = F.one_hot(source_phase, num_classes=7).type(torch.FloatTensor).to(DEVICE)          # pS shape:  torch.Size([1, 256, 7])
            iS = F.one_hot(source_intervention, num_classes=8).type(torch.FloatTensor).to(DEVICE)  # iS shape:  torch.Size([1, 256, 8])
            pS = self.ReshapeDown(pS)   # pS shape:  torch.Size([1, 1, 7])
            iS = self.ReshapeDown(iS)   # iS shape:  torch.Size([1, 1, 12])         
            input = torch.cat([input, pS, iS], 2)   # input shape:  torch.Size([1, 1, 271])
            input = self.DownFCLayer(input)    # input shape:  torch.Size([1, 1, 256])

        x1 = self.down_conv1(input)   
        x2 = self.maxpool(x1) 
        x3 = self.down_conv2(x2)  
        x4 = self.maxpool(x3) 
        x5 = self.down_conv3(x4) 
        x6 = self.maxpool(x5) 
        x7 = self.down_conv4(x6)     

        # upsampling
        if self.Bottleneck:
            # one hot encoding
            pT = F.one_hot(target_phase, num_classes=7).type(torch.FloatTensor).to(DEVICE)  # pT.shape = torch.Size([1, 256, 7])
            iT = F.one_hot(target_intervention, num_classes=8).type(torch.FloatTensor).to(DEVICE) # iT.shape = torch.Size([1, 256, 8])
            x7 = torch.cat([x7, pT, iT], 2) # x7.shape = torch.Size([1, 256, 47])
            x7 = self.BottleNeckFCLinear(x7)  # x7 = torch.Size([1, 256, 32])
           
        
        x = self.up_trans1(x7)
        x = self.up_conv1(x)  
        x = self.up_trans2(x)
        x = self.up_conv2(x)  
        x = self.up_trans3(x)
        x = self.up_conv3(x)  
        x = self.out(x)
        return x



class ResNetGenerator(nn.Module):
    def __init__(self, INPUTCHANNELS, OUTPUTCHANNELS, blocks=6):
        super(ResNetGenerator, self).__init__()
        '''
        A Basic Unet Generator without skip connections
        '''
        self.blocks = blocks

        self.maxpool = nn.MaxPool1d((2))  

        self.down_conv1 = single_conv_pad(INPUTCHANNELS, 64) 
        self.down_conv2 = single_conv_pad(64, 128)
        self.down_conv3 = single_conv_pad(128, 256)

        self.residual1 = residual_block(256, 256)
        self.residual2 = residual_block(256, 256)
        self.residual3 = residual_block(256, 256)
        self.residual4 = residual_block(256, 256)
        self.residual5 = residual_block(256, 256)
        self.residual6 = residual_block(256, 256)
        self.residual7 = residual_block(256, 256)
        self.residual8 = residual_block(256, 256)
        self.residual9 = residual_block(256, 256)

        self.up_trans1 = nn.ConvTranspose1d(256, 128, kernel_size=(2), stride=2, padding=0)
        self.up_conv1 = single_conv_pad(128, 128)
        self.up_trans2 = nn.ConvTranspose1d(128, 64, kernel_size=(2), stride=2, padding=0)
        self.up_conv2 = single_conv_pad(64, 64)

        self.out = nn.Conv1d(64, OUTPUTCHANNELS, kernel_size=1) # kernel_size must be == 1

    def forward(self, input):
        # [Batch size, Channels in, Height, Width]
        x1 = self.down_conv1(input) 
        x2 = self.maxpool(x1) 
        x3 = self.down_conv2(x2)
        x4 = self.maxpool(x3) 
        x5 = self.down_conv3(x4) 

        # residual blocks
        x5 = x5 + self.residual1(x5)
        if self.blocks > 1:
            x5 = x5 + self.residual2(x5)
            x5 = x5 + self.residual3(x5)
        if self.blocks > 3:
            x5 = x5 + self.residual4(x5)
            x5 = x5 + self.residual5(x5)
            x5 = x5 + self.residual6(x5)           
        if self.blocks > 6:
            x5 = x5 + self.residual7(x5)
            x5 = x5 + self.residual8(x5)
            x5 = x5 + self.residual9(x5)  # x5.shape =  torch.Size([1, 256, 64])

        # # decoder
        x = self.up_trans1(x5)
        x = self.up_conv1(x)
        x = self.up_trans2(x)
        x = self.up_conv2(x)
        x = self.out(x)

        return x



class OneHotResNetGenerator(nn.Module):
    def __init__(self, INPUTCHANNELS, OUTPUTCHANNELS, WINDOWSIZE, blocks=6, Down = False, Bottleneck = True):
        super(OneHotResNetGenerator, self).__init__()
        '''
        A Basic Unet Generator without skip connections
        '''
        self.WINDOWSIZE = WINDOWSIZE
        self.blocks = blocks
        self.Down = Down
        self.Bottleneck = Bottleneck

        self.maxpool = nn.MaxPool1d((2))  

        self.ReshapeDown = conv_layer(256, 1)
        self.DownFCLayer = nn.Linear(275, 256)

        self.down_conv1 = double_conv_pad(INPUTCHANNELS, 64) 
        self.down_conv2 = double_conv_pad(64, 128)
        self.down_conv3 = double_conv_pad(128, 256)

        self.residual1 = residual_block(256, 256)
        self.residual2 = residual_block(256, 256)
        self.residual3 = residual_block(256, 256)
        self.residual4 = residual_block(256, 256)
        self.residual5 = residual_block(256, 256)
        self.residual6 = residual_block(256, 256)
        self.residual7 = residual_block(256, 256)
        self.residual8 = residual_block(256, 256)
        self.residual9 = residual_block(256, 256)

        self.BottleNeckFCLinear = nn.Linear(79, 64)

        self.up_trans1 = nn.ConvTranspose1d(256, 128, kernel_size=(2), stride=2, padding=0)
        self.up_conv1 = double_conv_pad(128, 128)
        self.up_trans2 = nn.ConvTranspose1d(128, 64, kernel_size=(2), stride=2, padding=0)
        self.up_conv2 = double_conv_pad(64, 64)

        self.out = nn.Conv1d(64, OUTPUTCHANNELS, kernel_size=1) # kernel_size must be == 1

    def forward(self, input, source_phase, source_intervention, target_phase, target_intervention):

        if self.Down:  # input shape:  torch.Size([1, 1, 256]) 
            pS = F.one_hot(source_phase, num_classes=7).type(torch.FloatTensor).to(DEVICE)          # pS shape:  torch.Size([1, 256, 7])
            iS = F.one_hot(source_intervention, num_classes=8).type(torch.FloatTensor).to(DEVICE)  # iS shape:  torch.Size([1, 256, 8])
            pS = self.ReshapeDown(pS)   # pS shape:  torch.Size([1, 1, 7])
            iS = self.ReshapeDown(iS)   # iS shape:  torch.Size([1, 1, 12])         
            input = torch.cat([input, pS, iS], 2)   # input shape:  torch.Size([1, 1, 275])
            input = self.DownFCLayer(input)    # input shape:  torch.Size([1, 1, 256])

        x1 = self.down_conv1(input) 
        x2 = self.maxpool(x1) 
        x3 = self.down_conv2(x2)
        x4 = self.maxpool(x3) 
        x5 = self.down_conv3(x4)  

        # residual blocks
        x5 = x5 + self.residual1(x5)
        x5 = x5 + self.residual2(x5)
        x5 = x5 + self.residual3(x5)
        if self.blocks > 3:
            x5 = x5 + self.residual4(x5)
            x5 = x5 + self.residual5(x5)
            x5 = x5 + self.residual6(x5)           
        if self.blocks > 6:
            x5 = x5 + self.residual7(x5)
            x5 = x5 + self.residual8(x5)
            x5 = x5 + self.residual9(x5)  # x5.shape =  torch.Size([1, 256, 64])

        # upsampling
        if self.Bottleneck:
            # one hot encoding 
            pT = F.one_hot(target_phase, num_classes=7).type(torch.FloatTensor).to(DEVICE)  #     pT.shape =  torch.Size([1, 256, 7])
            iT = F.one_hot(target_intervention, num_classes=8).type(torch.FloatTensor).to(DEVICE) #     iT.shape =  torch.Size([1, 256, 8])
            x5 = torch.cat([x5, pT, iT], 2)    # x5.shape =  torch.Size([1, 256, 79])
            x5 = self.BottleNeckFCLinear(x5)   # x5.shape =  torch.Size([1, 256, 64])

        # # decoder
        x = self.up_trans1(x5)
        x = self.up_conv1(x)
        x = self.up_trans2(x)
        x = self.up_conv2(x)
        x = self.out(x)

        return x  