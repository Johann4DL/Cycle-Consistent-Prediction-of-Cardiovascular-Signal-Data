import torch
from torch.utils.data import Dataset
import numpy as np

class UnpairedEmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_names, target_name, window_length=256):
        self.df = df
        self.feature_names = feature_names
        self.target_name = target_name
        self.window_length = window_length
        
        self.num_animals = len(np.unique(df["animal"]))
        self.animal_dfs = [group[1] for group in df.groupby("animal")]
        # get statistics for test dataset
        self.animal_lens = [len(an_df) // self.window_length for an_df in self.animal_dfs]
        self.animal_cumsum = np.cumsum(self.animal_lens)
        self.num_windows = sum(self.animal_lens)

    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):
    
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
        phase_df_source = animal_df_source["Phasenzuordnung"]   
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

        return inputs, targets, phase_source, intervention_source, phase_target, intervention_target




class AnimalDatasetEmbedding(torch.utils.data.Dataset):
    def __init__(self, df, feature_names, target_name, 
                 window_length=256):
        self.df = df
        self.feature_names = feature_names
        self.target_name = target_name
        self.window_length = window_length
        
        self.num_animals = len(np.unique(df["animal"]))
        self.animal_dfs = [group[1] for group in df.groupby("animal")]  # list of animal dfs
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
        
        animal_idx = int(np.where(self.animal_cumsum >= idx)[0][0])
        animal_df = self.animal_dfs[animal_idx]
        
        # take window
        start_idx = np.random.randint(0, len(animal_df) - self.window_length - 1)
        end_idx = start_idx + self.window_length
        animal_df = animal_df.iloc[start_idx: end_idx]  
        
        # extract features
        input_df = animal_df[self.feature_names]
        target_df = animal_df[self.target_name]
        phase_df = animal_df["Phasenzuordnung"]   
        intervention_df = animal_df["intervention"]
        
        # to torch
        inputs = torch.tensor(input_df.to_numpy()).permute(1, 0)
        targets = torch.tensor(target_df.to_numpy()).unsqueeze(0)
        phase = torch.tensor(phase_df.to_numpy()).type(torch.LongTensor)
        intervention = torch.tensor(intervention_df.to_numpy()).type(torch.LongTensor)
        
        # the phase and intervention are the same for source and target
        return inputs, targets, phase, intervention, phase, intervention




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

        # to torch
        inputs = torch.tensor(input_df.to_numpy()).permute(1, 0)
        targets = torch.tensor(target_df.to_numpy()).unsqueeze(0)

        return inputs, targets



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




class UnequalDataSetSize(torch.utils.data.Dataset):
    def __init__(self, df, target_df, feature_names, target_name, window_length=256):
        self.df = df
        self.feature_names = feature_names
        self.target_name = target_name
        self.window_length = window_length
        self.target_df = target_df
        
        self.num_animals = len(np.unique(df["animal"]))
        self.animal_dfs = [group[1] for group in df.groupby("animal")]
        # get statistics for test dataset
        self.animal_lens = [len(an_df) // self.window_length for an_df in self.animal_dfs]
        self.animal_cumsum = np.cumsum(self.animal_lens)
        self.num_windows = sum(self.animal_lens)
     

    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):

        animal_idx = int(np.where(self.animal_cumsum >= idx)[0][0])
        animal_df = self.animal_dfs[animal_idx]

        # take different windows for the source and target -> unpaired examples
        start_idx_source = np.random.randint(0, len(animal_df) - self.window_length - 1)
        start_idx_target = np.random.randint(0, len(self.target_df) - self.window_length - 1)
        end_idx_source = start_idx_source + self.window_length
        end_idx_target = start_idx_target + self.window_length
        animal_df_source = animal_df.iloc[start_idx_source: end_idx_source]
        animal_df_target = self.target_df.iloc[start_idx_target: end_idx_target]

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

        return inputs, targets, phase_source, intervention_source, phase_target, intervention_target
    



