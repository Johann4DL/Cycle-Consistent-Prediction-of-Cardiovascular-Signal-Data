import torch
import torch.nn as nn
import config
import pandas as pd
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import utils



def save_checkpoint(model, optimizer, path="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint at location: ", path)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)



def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint from location: ", checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr



def save_predictions(loader, gen_A2B, gen_B2A, DEVICE):
    for sig_A, sig_B in loader:
        #convert to float16
        sig_A = sig_A.float()
        sig_B = sig_B.float()
        #move to GPU
        sig_A = sig_A.to(DEVICE)
        sig_B = sig_B.to(DEVICE)

        fake_B = gen_A2B(sig_A)
        fake_A = gen_B2A(sig_B)

        #reshape to 1D
        fake_B = fake_B.reshape(-1)
        fake_A = fake_A.reshape(-1)
        sig_A = sig_A.reshape(-1)
        sig_B = sig_B.reshape(-1)

        #save generated signals as csv in one file
        df = pd.DataFrame({'sig_A': sig_A.cpu().numpy(), 'fake_A': fake_A.cpu().numpy(), 
                        'sig_B': sig_B.cpu().numpy(), 'fake_B': fake_B.cpu().numpy()})
        df.to_csv('generated_signals.csv', index=True)



def load_csv(path):
    '''
    Load the data from a csv and return a dataframe
    '''
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, sep=";")
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame



def drop_cols(df):
    '''
    Drop the columns that are not relevant to the task
    '''
    df = df.drop(columns=['Time',  'RVP', 'PaP', 'PaQ', 'ECGcond', 'Looperkennung', 'Extrasystolen', 'Ansaugphase', 'ECG', 'ECGcond', 'LVV1', 'LVV2', 
                      'LVV3', 'LVV4', 'LVV5', 'RVV1', 'RVV2', 'RVV3', 'RVV4', 'RVV5', 'Versuchsdatum', 'rep_an', 'rep_sect', #'contractility', 
                      'preload', 'afterload', 'controller'])  # 'VADspeed', 'LVtot', 'RVtot', 'AoQ'
    return df



def remove_strings(df):
    '''
    Removes all strings that remain after dropping all unrelevant columns in the data. Works only if you called 'drop_cols' first
    '''

    df_cols = ['LVtot_kalibriert', 'LVP', 'AoP', 'AoQ', 'RVtot_kalibriert', 'VADspeed', 'VadQ', 'VADcurrent', 'Phasenzuordnung', 
            'LVtot', 'RVtot', 'animal', 'contractility', 'intervention']

    df = df.to_numpy() # convert to numpy

    # iterate over rows and columns of numpy array
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            if type(df[row, col]) == str:
                df[row, col] = df[row, col].replace('[ml]', '')
                df[row, col] = df[row, col].replace('[mmHg]', '')
                df[row, col] = df[row, col].replace('[L/min]', '')
                df[row, col] = df[row, col].replace('[rpm]', '')
                df[row, col] = df[row, col].replace('[A]', '')
                df[row, col] = df[row, col].replace('[mL]', '')
                df[row, col] = float(df[row, col])

    df = pd.DataFrame(df, columns=df_cols) # convert back to dataframe
    return df



def groupedAvg(myArray, N=10):
    '''
    Subsamples an array by N
    '''
    result = np.cumsum(myArray, 0)[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]
    return result



def subsample(df, rate):
    '''
    Subsample a dataframe by the rate using 'groupedAvg'
    '''
    columns = ['LVtot_kalibriert', 'LVP', 'AoP', 'AoQ', 'RVtot_kalibriert', 'VADspeed', 'VadQ', 
            'VADcurrent', 'Phasenzuordnung', 'LVtot', 'RVtot', 'animal', 'contractility', 'intervention']   

    arr = df.to_numpy()
    arr = groupedAvg(arr, N=rate)
    df = pd.DataFrame(arr, columns=columns)
    df['animal'] = df['animal'].astype(int)
    df['Phasenzuordnung'] = df['Phasenzuordnung'].astype(int)
    df['intervention'] = df['intervention'].astype(int)
    df['intervention'] = df['intervention'].astype(float)
    return df



def visualize(df, variables, length):
    fig, axs = plt.subplots(len(variables), 1, figsize=(20, 10))
    axs[0].plot(df[variables[0]][:length])
    axs[0].set_title(variables[0])
    axs[1].plot(df[variables[1]][:length])
    axs[1].set_title(variables[1])
    axs[2].plot(df[variables[2]][:length])
    axs[2].set_title(variables[2])
    axs[3].plot(df[variables[3]][:length])
    axs[3].set_title(variables[3])
    axs[4].plot(df[variables[4]][:length])
    axs[4].set_title(variables[4])
    axs[5].plot(df[variables[5]][:length])
    axs[5].set_title(variables[5])
    axs[6].plot(df[variables[6]][:length])
    axs[6].set_title(variables[6])
    plt.show()



def get_data_overview(df):
    print('Shape of DataFrame', df.shape)
    print('AoP: mean: ', df["AoP" ].mean(), 'std: ', df["AoP" ].std(), 'min: ', df["AoP" ].min(), 'max: ', df["AoP" ].max(), 'median: ', df["AoP" ].median())
    print('VADcurrent: mean: ', df["VADcurrent" ].mean(), 'std: ', df["VADcurrent" ].std(), 'min: ', df["VADcurrent" ].min(), 'max: ', df["VADcurrent" ].max(), 'median: ', df["VADcurrent" ].median())
    print('VadQ: mean: ', df["VadQ" ].mean(), 'std: ', df["VadQ" ].std(), 'min: ', df["VadQ" ].min(), 'max: ', df["VadQ" ].max(), 'median: ', df["VadQ" ].median())
    print('LVP: mean: ', df["LVP" ].mean(), 'std: ', df["LVP" ].std(), 'min: ', df["LVP" ].min(), 'max: ', df["LVP" ].max(), 'median: ', df["LVP" ].median())
    print('LVtot_kalibriert: mean: ', df["LVtot_kalibriert" ].mean(), 'std: ', df["LVtot_kalibriert" ].std(), 'min: ', df["LVtot_kalibriert" ].min(), 'max: ', df["LVtot_kalibriert" ].max(), 'median: ', df["LVtot_kalibriert" ].median())



def normalize_by_all_phases(df, scaler):
    '''
    Normalize the data by the whole dataframe
    '''
    cols = df.columns.tolist()
    df = df.to_numpy() 
    scaler.fit(df)
    transformed_data = scaler.transform(df)
    df = pd.DataFrame(transformed_data, columns=cols)  
    return df



def normalize_by_phase1(df, scaler):
    '''
    Normalize the data by the first phase
    '''
    cols = df.columns.tolist()
    phase_1 = df.loc[df['Phasenzuordnung'] == 1]
    phase_1 = phase_1.to_numpy() 
    df = df.to_numpy()  
    scaler.fit(phase_1)
    transformed_data = scaler.transform(df)
    df = pd.DataFrame(transformed_data, columns=cols)  
    return df



def normalize(df, scaler, phase1 = True):
    df_IPA = df[['intervention', 'Phasenzuordnung', 'animal','contractility']]
    df_temp = pd.DataFrame()
    NORMALIZE = utils.normalize_by_phase1
    if phase1 == True:
        NORMALIZE = utils.normalize_by_all_phases

    for animal in df['animal'].unique():
        # split df into separate dataframes for each animal
        df_animal = df.loc[df['animal'] == animal]
        df_animal = NORMALIZE(df_animal, scaler) # normalize by phase 1
        # append df_animal to df_temp
        df_temp = pd.concat([df_temp, df_animal], axis=0, ignore_index=True)

    df = df_temp
    df = df.drop(columns=['intervention', 'Phasenzuordnung', 'animal', 'contractility'])
    df.dropna(inplace=True)
    df = df.join(df_IPA)
    return df



def gen_signals(fake_target, fake_source, target, source):
    fake_target = fake_target.reshape(-1)
    fake_source = fake_source.reshape(-1)
    source = source.reshape(-1)
    target = target.reshape(-1)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(source.cpu().detach().numpy(), label= 'Real source signals')
    ax[0].plot(fake_source.cpu().detach().numpy(), label= 'Recreated source signals')
    ax[0].set_xlabel('Signal length')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(target.cpu().detach().numpy(), label= 'Real target signal')
    ax[1].plot(fake_target.cpu().detach().numpy(), label= 'Recreated target signal')
    ax[1].set_xlabel('Signal length')
    ax[1].set_ylabel('Loss')
    ax[1].legend()



def discriminator_loss(disc, reals, fakes):
    # calculate how close reals are to being classified as real
    real_loss = nn.MSELoss(disc(reals), torch.ones_like(disc(reals)))
    # calculate how close fakes are to being classified as fake
    fake_loss = nn.MSELoss(disc(fakes), torch.zeros_like(disc(fakes)))
    # return the average of real and fake loss
    return (real_loss + fake_loss) / 2



# @torch.cuda.amp.autocast()
def get_disc_loss(source, target, disc_source, disc_target, fake_source, fake_target
                    ):
    """
    Return the loss of the discriminator given inputs.
    """
    
    # discriminator loss
    disc_target_loss = utils.discriminator_loss(disc_target, target, fake_target)
    disc_source_loss = utils.discriminator_loss(disc_source, source, fake_source)
    disc_loss = (disc_source_loss + disc_target_loss) #/ 2

    return disc_loss, disc_source_loss, disc_target_loss
