import torch
import config
import pandas as pd
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
                      'LVV3', 'LVV4', 'LVV5', 'RVV1', 'RVV2', 'RVV3', 'RVV4', 'RVV5', 'Versuchsdatum', 'rep_an', 'rep_sect', 'contractility', 
                      'preload', 'afterload', 'controller'])
    return df


def remove_strings(df):
    '''
    Removes all strings that remain after dropping all unrelevant columns in the data. Works only if you called 'drop_cols' first
    '''

    df_cols = df_cols = ['LVtot_kalibriert', 'LVP', 'AoP', 'AoQ', 'RVtot_kalibriert', 'VADspeed', 'VadQ', 'VADcurrent', 'Phasenzuordnung', 
            'LVtot', 'RVtot', 'animal', 'intervention']

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
            'VADcurrent', 'Phasenzuordnung', 'LVtot', 'RVtot', 'animal', 'intervention']

    arr = df.to_numpy()
    arr = groupedAvg(arr, N=rate)
    df = pd.DataFrame(arr, columns=columns)
    df['animal'] = df['animal'].astype(int)
    df['intervention'] = df['intervention'].astype(int)
    return df

def normalize(df):
    # intervntion, Phasenzuordnung and animal should be in another dataframe before the data is normalized
    df_IPA = df[['intervention', 'Phasenzuordnung', 'animal']]
    df = df.drop(columns=['intervention', 'Phasenzuordnung', 'animal'])  # drop columns in original dataframe

    df_cols = ['LVtot_kalibriert', 'LVP', 'AoP', 'AoQ', 'RVtot_kalibriert', 'VADspeed', 'VadQ', 'VADcurrent', 'LVtot', 'RVtot']
    df = df.to_numpy()  #convert to numpy

    # scale the data
    scaler = StandardScaler()
    scaler.fit(df)
    transformed_data = scaler.transform(df)
    df = pd.DataFrame(transformed_data, columns=df_cols)  # convert to dataframe
    df = pd.concat([df, df_IPA], axis=1) # add the drpped columns again
    return df

def visualize(df, var1, var2, var3, var4, length):
    fig, axs = plt.subplots(4, 1, figsize=(20, 10))
    axs[0].plot(df[var1][:length])
    axs[0].set_title(var1)
    axs[1].plot(df[var2][:length])
    axs[1].set_title(var2)
    axs[2].plot(df[var3][:length])
    axs[2].set_title(var3)
    axs[3].plot(df[var4][:length])
    axs[3].set_title(var4)
    plt.show()



