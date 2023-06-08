import torch
from torch.utils.data import Dataset
import numpy as np


class UnpairedDataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_names, target_name, test, window_length=256):
        self.df = df
        self.feature_names = feature_names
        self.target_name = target_name
        self.window_length = window_length
        self.test = test
        
        self.num_animals = len(np.unique(df["animal"]))
        self.animal_dfs = [group[1] for group in df.groupby("animal")]
        # get statistics for test dataset
        self.animal_lens = [len(an_df) // self.window_length for an_df in self.animal_dfs]
        self.animal_cumsum = np.cumsum(self.animal_lens)
        self.num_windows = sum(self.animal_lens)

    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):
        if self.test:
            # look up which test animal the idx corresponds to
            animal_idx = int(np.where(self.animal_cumsum >= idx)[0][0])
            animal_df = self.animal_dfs[animal_idx]
            # look up which part of the test animal the idx corresponds to 
            if animal_idx > 0:
                start_idx_source = idx - self.animal_cumsum[animal_idx - 1]
                start_idx_target = idx - self.animal_cumsum[animal_idx - 1]
            else:
                start_idx_source = idx
                start_idx_target = idx
            start_idx_source *= self.window_length
            start_idx_target *= self.window_length
        else:
            animal_idx = int(np.where(self.animal_cumsum >= idx)[0][0])
            animal_df = self.animal_dfs[animal_idx]

            # take different windows for the source and target -> unpaired examples
            start_idx_source = np.random.randint(0, len(animal_df) - self.window_length - 1)
            start_idx_target = np.random.randint(0, len(animal_df) - self.window_length - 1)
        end_idx_source = start_idx_source + self.window_length
        end_idx_target = start_idx_target + self.window_length
        animal_df_source = animal_df.iloc[start_idx_source: end_idx_source]
        animal_df_target = animal_df.iloc[start_idx_target: end_idx_target]

        # extract features
        input_df = animal_df_source[self.feature_names]
        target_df = animal_df_target[self.target_name]
        phase_df_source = animal_df_source["Phasenzuordnung"]   # mit 'Phasenzuordnung' klappt es
        intervention_df_source = animal_df_source["intervention"]
        phase_df_target = animal_df_target["Phasenzuordnung"]
        intervention_df_target = animal_df_target["intervention"]

        # to torch
        inputs = torch.tensor(input_df.to_numpy()).permute(1, 0)
        targets = torch.tensor(target_df.to_numpy()).unsqueeze(0)
        phase_source = torch.tensor(phase_df_source.to_numpy()).type(torch.LongTensor)
        intervention_source = torch.tensor(intervention_df_source.to_numpy()).type(torch.LongTensor)
        phase_target = torch.tensor(phase_df_target.to_numpy()).type(torch.LongTensor)
        intervention_target = torch.tensor(intervention_df_target.to_numpy()).type(torch.LongTensor)

        return inputs, targets, phase_source, intervention_source #, phase_target, intervention_target 


class AnimalDataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_names, target_name, test,
                 window_length=256):
        self.df = df
        self.feature_names = feature_names
        self.target_name = target_name
        self.window_length = window_length
        self.test = test
        
        self.num_animals = len(np.unique(df["animal"]))
        self.animal_dfs = [group[1] for group in df.groupby("animal")]
        # get statistics for test dataset
        self.animal_lens = [len(an_df) // self.window_length for an_df in self.animal_dfs]
        self.animal_cumsum = np.cumsum(self.animal_lens)
        self.num_windows = sum(self.animal_lens)

        
    def __len__(self):
        # if self.test:
        return self.num_windows
        # else:
        #     return self.num_animals
    
    def __getitem__(self, idx):
        if self.test:
            # look up which test animal the idx corresponds to
            animal_idx = int(np.where(self.animal_cumsum >= idx)[0][0])
            animal_df = self.animal_dfs[animal_idx]
            # look up which part of the test animal the idx corresponds to 
            if animal_idx > 0:
                start_idx = idx - self.animal_cumsum[animal_idx - 1]
            else:
                start_idx = idx
            start_idx *= self.window_length
        else:
            # animal_df = self.animal_dfs[idx]
            animal_idx = int(np.where(self.animal_cumsum >= idx)[0][0])
            animal_df = self.animal_dfs[animal_idx]
            
            # take window
            start_idx = np.random.randint(0, len(animal_df) - self.window_length - 1)
        end_idx = start_idx + self.window_length
        animal_df = animal_df.iloc[start_idx: end_idx]
        
        # extract features
        input_df = animal_df[self.feature_names]
        target_df = animal_df[self.target_name]
        
        # to torch
        inputs = torch.tensor(input_df.to_numpy()).permute(1, 0)
        targets = torch.tensor(target_df.to_numpy()).unsqueeze(0)
        
        return inputs, targets
    

class AnimalDatasetEmbedding(torch.utils.data.Dataset):
    def __init__(self, df, feature_names, target_name, test,
                 window_length=256):
        self.df = df
        self.feature_names = feature_names
        self.target_name = target_name
        self.window_length = window_length
        self.test = test
        
        self.num_animals = len(np.unique(df["animal"]))
        self.animal_dfs = [group[1] for group in df.groupby("animal")]
        # get statistics for test dataset
        self.animal_lens = [len(an_df) // self.window_length for an_df in self.animal_dfs]
        self.animal_cumsum = np.cumsum(self.animal_lens)
        self.num_windows = sum(self.animal_lens)

        
    def __len__(self):
        # if self.test:
        return self.num_windows
        # else:
        #     return self.num_animals
    
    def __getitem__(self, idx):
        if self.test:
            # look up which test animal the idx corresponds to
            animal_idx = int(np.where(self.animal_cumsum >= idx)[0][0])
            animal_df = self.animal_dfs[animal_idx]
            # look up which part of the test animal the idx corresponds to 
            if animal_idx > 0:
                start_idx = idx - self.animal_cumsum[animal_idx - 1]
            else:
                start_idx = idx
            start_idx *= self.window_length
        else:
            # animal_df = self.animal_dfs[idx]
            animal_idx = int(np.where(self.animal_cumsum >= idx)[0][0])
            animal_df = self.animal_dfs[animal_idx]
            
            # take window
            start_idx = np.random.randint(0, len(animal_df) - self.window_length - 1)
        end_idx = start_idx + self.window_length
        animal_df = animal_df.iloc[start_idx: end_idx]
        
        # extract features
        input_df = animal_df[self.feature_names]
        target_df = animal_df[self.target_name]
        phase_df = animal_df["Phasenzuordnung"]   # mit 'Phasenzuordnung' klappt es
        intervention_df = animal_df["intervention"]
        
        # to torch
        inputs = torch.tensor(input_df.to_numpy()).permute(1, 0)
        targets = torch.tensor(target_df.to_numpy()).unsqueeze(0)
        phase = torch.tensor(phase_df.to_numpy()).type(torch.LongTensor)
        intervention = torch.tensor(intervention_df.to_numpy()).type(torch.LongTensor)
        
        return inputs, targets, phase, intervention

class TrainDataset(Dataset):
    def __init__(self, signal_A, signal_B, df):
        self.df = df
        self.signal_A = self.df[signal_A]
        self.signal_B = self.df[signal_B]

        # only data from a single animal per batch 
        for animal in self.df['animal'].unique():
            df_single_animal = self.df[self.df['animal'] == animal]
            
            # length should be modulo 256 = 0
            df_single_animal = df_single_animal.iloc[:-(len(df_single_animal) % 256), :]
            # creating tensor from df 
            tensor_A = torch.tensor(df_single_animal[signal_A].values)
            tensor_B = torch.tensor(df_single_animal[signal_B].values)

            # split tensor into tensors of size 256
            tensor_A = tensor_A.split(256)  # tensor shape (256, 1) 
            tensor_B = tensor_B.split(256)       

            # stack tensors
            stack_A = torch.stack(tensor_A).unsqueeze(1) 
            stack_B = torch.stack(tensor_B).unsqueeze(1) 
            #print(stack_A.shape, stack_B.shape)
            
            if animal == self.df['animal'].unique()[0]:
                # add stack
                self.tensor_A = stack_A
                self.tensor_B = stack_B
                
            else:
                # add stack
                self.tensor_A = torch.cat((self.tensor_A, stack_A), 0)
                self.tensor_B = torch.cat((self.tensor_B, stack_B), 0)
          

    def __len__(self):
        # signal_A and signal_B should have the same length
        return len(self.tensor_A)

    def __getitem__(self, index):
        # return the signal at the given index  # add data augmentation?
        return self.tensor_A[index], self.tensor_B[index]
    

class TestDataset(Dataset):
    def __init__(self, signal_A, signal_B, df):
        self.df = df
        self.signal_A = self.df[signal_A]
        self.signal_B = self.df[signal_B]

        # length should be modulo 256 = 0
        self.df = self.df.iloc[:-(len(self.df) % 256), :]
        
        # creating tensor from df 
        tensor_A = torch.tensor(self.df[signal_A].values)
        tensor_B = torch.tensor(self.df[signal_B].values)

        # split tensor into tensors of size 256
        tensor_A = tensor_A.split(256)  # tensor shape (256, 1) 
        tensor_B = tensor_B.split(256)       

        # stack tensors
        self.tensor_A = torch.stack(tensor_A).unsqueeze(1) 
        self.tensor_B = torch.stack(tensor_B).unsqueeze(1) 
        # print(self.tensor_A.shape, self.tensor_A.shape)


    def __len__(self):
        # signal_A and signal_B should have the same length
        return len(self.tensor_A)

    def __getitem__(self, index):
        # return the signal at the given index  # add data augmentation?
        return self.tensor_A[index], self.tensor_B[index]