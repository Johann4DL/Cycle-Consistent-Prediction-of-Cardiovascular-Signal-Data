import torch
import torch.nn as nn

def double_conv_pad(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='zeros'),
        # nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Dropout1d(p=0.1, inplace=False),
        nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='zeros'),
        # nn.BatchNorm1d(out_channels),
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
        self.down_conv5 = double_conv_pad(256, 512)
        self.down_conv6 = double_conv_pad(512, 1024)

        self.embedding = torch.nn.Embedding(num_embeddings=11, embedding_dim=32)  # if num_embeddings < 11 I get an error, even though
                                                                                  # there are only 9 different interventions
        

        self.up_trans1 = nn.ConvTranspose1d(1024, 512, kernel_size=(2), stride=2, padding=0)
        self.up_conv1 = double_conv_pad(1024, 512)
        self.up_trans2 = nn.ConvTranspose1d(512, 256, kernel_size=(2), stride=2, padding=0)
        self.up_conv2 = double_conv_pad(512, 256)
        self.up_trans3 = nn.ConvTranspose1d(256, 128, kernel_size=(2), stride=2, padding=0)
        self.up_conv3 = double_conv_pad(256, 128)
        self.up_trans4 = nn.ConvTranspose1d(128, 64, kernel_size=(2), stride=2, padding=0)
        self.up_conv4 = double_conv_pad(128, 64)
        self.up_trans5 = nn.ConvTranspose1d(64, 32, kernel_size=(2), stride=2, padding=0)
        self.up_conv5 = double_conv_pad(64, 32)

        self.out = nn.Conv1d(32, OUTPUTCHANNELS, kernel_size=1) # kernel_size must be == 1

    def forward(self, input, phase, intervention):
        # [Batch size, Channels in, Height, Width]
        
        # downsampling
        x1 = self.down_conv1(input)   
        x2 = self.maxpool(x1) 
        x3 = self.down_conv2(x2)  
        x4 = self.maxpool(x3) 
        x5 = self.down_conv3(x4) 
        x6 = self.maxpool(x5) 
        x7 = self.down_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.down_conv5(x8)
        x10 = self.maxpool(x9)
        x11 = self.down_conv6(x10)

        # upsampling
        p = self.embedding(phase)
        # p = torch.cat([p, p, p, p], 1)
        i = self.embedding(intervention)
        # i = torch.cat([i, i, i, i], 1)
        x7 = x7 + p + i                               # The phase embedding is added during upsampling
        x = self.up_trans1(x11)
        x = self.up_conv1(torch.cat([x, x9], 1))  # skip connection
        x = self.up_trans2(x)
        x = self.up_conv2(torch.cat([x, x7], 1))  # skip connection
        x = self.up_trans3(x)
        x = self.up_conv3(torch.cat([x, x5], 1))  # skip connection
        x = self.up_trans4(x)
        x = self.up_conv4(torch.cat([x, x3], 1))  # skip connection
        x = self.up_trans5(x)
        x = self.up_conv5(torch.cat([x, x1], 1))  # skip connection
        x = self.out(x)
        return x
    


class EmbeddingEncoderDecoderGenerator(nn.Module):
    def __init__(self, INPUTCHANNELS, OUTPUTCHANNELS):
        super(EmbeddingEncoderDecoderGenerator, self).__init__()
        self.maxpool = nn.MaxPool1d((2))  

        self.down_conv1 = double_conv_pad(INPUTCHANNELS, 32) 
        self.down_conv2 = double_conv_pad(32, 64) 
        self.down_conv3 = double_conv_pad(64, 128)
        self.down_conv4 = double_conv_pad(128, 256)
        self.down_conv5 = double_conv_pad(256, 512)
        self.down_conv6 = double_conv_pad(512, 1024)

        self.embedding = torch.nn.Embedding(num_embeddings=11, embedding_dim=32)  # if num_embeddings < 11 I get an error, even though
                                                                                  # there are only 9 different interventions
        
        self.up_trans1 = nn.ConvTranspose1d(1024, 512, kernel_size=(2), stride=2, padding=0)
        self.up_conv1 = double_conv_pad(512, 512)
        self.up_trans2 = nn.ConvTranspose1d(512, 256, kernel_size=(2), stride=2, padding=0)
        self.up_conv2 = double_conv_pad(256, 256)
        self.up_trans3 = nn.ConvTranspose1d(256, 128, kernel_size=(2), stride=2, padding=0)
        self.up_conv3 = double_conv_pad(128, 128)
        self.up_trans4 = nn.ConvTranspose1d(128, 64, kernel_size=(2), stride=2, padding=0)
        self.up_conv4 = double_conv_pad(64, 64)
        self.up_trans5 = nn.ConvTranspose1d(64, 32, kernel_size=(2), stride=2, padding=0)
        self.up_conv5 = double_conv_pad(32, 32)

        self.out = nn.Conv1d(32, OUTPUTCHANNELS, kernel_size=1) # kernel_size must be == 1

    def forward(self, input, phase, intervention):
        # [Batch size, Channels in, Height, Width]
        
        # downsampling
        x1 = self.down_conv1(input)   
        x2 = self.maxpool(x1) 
        x3 = self.down_conv2(x2)  
        x4 = self.maxpool(x3) 
        x5 = self.down_conv3(x4) 
        x6 = self.maxpool(x5) 
        x7 = self.down_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.down_conv5(x8)
        x10 = self.maxpool(x9)
        x11 = self.down_conv6(x10)

        # upsampling
        p = self.embedding(phase)
        i = self.embedding(intervention)
        x = self.up_trans1(x11)
        x = self.up_conv1(x)  
        x = self.up_trans2(x)
        x = x + p + i                           # The phase embedding is added during upsampling
        x = self.up_conv2(x)  # skip connection
        x = self.up_trans3(x)
        x = self.up_conv3(x)  # skip connection
        x = self.up_trans4(x)
        x = self.up_conv4(x)  # skip connection
        x = self.up_trans5(x)
        x = self.up_conv5(x)  # skip connection
        x = self.out(x)
        return x



class UnpairedEncoderDecoder(nn.Module):
    def __init__(self, INPUTCHANNELS, OUTPUTCHANNELS):
        super(UnpairedEncoderDecoder, self).__init__()
        self.maxpool = nn.MaxPool1d((2))  

        self.source_embedding = torch.nn.Embedding(num_embeddings=11, embedding_dim=256)
        self.down_conv1 = double_conv_pad(INPUTCHANNELS, 32) 
        self.down_conv2 = double_conv_pad(32, 64) 
        self.down_conv3 = double_conv_pad(64, 128)
        self.down_conv4 = double_conv_pad(128, 256)

        self.target_embedding = torch.nn.Embedding(num_embeddings=11, embedding_dim=32)
        self.up_trans1 = nn.ConvTranspose1d(256, 128, kernel_size=(2), stride=2, padding=0)
        self.up_conv1 = double_conv_pad(128, 128)
        self.up_trans2 = nn.ConvTranspose1d(128, 64, kernel_size=(2), stride=2, padding=0)
        self.up_conv2 = double_conv_pad(64, 64)
        self.up_trans3 = nn.ConvTranspose1d(64, 32, kernel_size=(2), stride=2, padding=0)
        self.up_conv3 = double_conv_pad(32, 32)

        self.out = nn.Conv1d(32, OUTPUTCHANNELS, kernel_size=1) # kernel_size must be == 1

    def forward(self, input, source_phase, source_intervention, target_phase, target_intervention):
        # [Batch size, Channels in, Height, Width]

        sp = self.source_embedding(source_phase)
        si = self.source_embedding(source_intervention)
        input += sp + si # source embeddings are added before downsampling
        x1 = self.down_conv1(input) 
        x2 = self.maxpool(x1) 
        x3 = self.down_conv2(x2)
        x4 = self.maxpool(x3) 
        x5 = self.down_conv3(x4) 
        x6 = self.maxpool(x5)  
        x7 = self.down_conv4(x6)

        # # decoder

        # verschiedene Experimente, embedding nur im bottle neck oder auch beim downsampling
        # one hot encoding oder als vektor?
        # vershciedene Experimente nacheinander rum zu schauen was welchen effekt hatte
        # in phase 1 keine intervention
        # verschiedene Mengen von Daten
        # stattdessen koente ich einfach ein embedding uebergeben. erst nur phase 3 und dann sukessiv mehr phasen und interventionen
        tp = self.target_embedding(target_phase)  # lieber one hot encoding ( 1, 0, 0 , 0)
        ti = self.target_embedding(target_intervention)
        x7 += tp + ti # target embeddings are added before upsampling  # lieber concatenating und dann fully connected to amtch the dimension
        x = self.up_trans1(x7)
        x = self.up_conv1(x)
        x = self.up_trans2(x)
        x = self.up_conv2(x)
        x = self.up_trans3(x)
        x = self.up_conv3(x)
        x = self.out(x)

        return x