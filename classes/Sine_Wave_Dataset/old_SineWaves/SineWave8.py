from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from classes.Sine_Wave_Dataset.helper_sine import add_noise

# L8: Changing Amplitude (20-80), Changing Period (40-100), Clock-like Input (resetting
# Add more noise to clock like input after each 200th epoch
class SineWaveDataset81(Dataset):
    decrease_vector = None
    next_vector = None

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.num_timesteps = 300

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        amplitude = self.data.iloc[idx, 0]
        period = self.data.iloc[idx, 1]
        sine_wave = eval(self.data.iloc[idx, 2])  # assuming the sine_wave is stored as a string representation of a list
        time_vector = torch.tensor([i for i in range(period)] * (self.num_timesteps // int(period) + 1), dtype=torch.float32)
        time_vector = time_vector[:self.num_timesteps]
        
        # Compute only if it's the first call
        if SineWaveDataset81.decrease_vector is None or SineWaveDataset81.next_vector is None:
            SineWaveDataset81.decrease_vector = time_vector / 5
            SineWaveDataset81.next_vector = add_noise(time_vector - SineWaveDataset81.decrease_vector)
        
        amplitude_vector = torch.tensor([amplitude] * self.num_timesteps, dtype=torch.float32)
        period_vector = torch.tensor([period] * self.num_timesteps, dtype=torch.float32)
        sine_wave_vector = torch.tensor(sine_wave, dtype=torch.float32)
        input_vector = torch.stack([amplitude_vector, period_vector, time_vector], dim=1)  
        # Shape: [num_timesteps, 2]
        target_vector = torch.stack([sine_wave_vector], dim=1)
        return input_vector, sine_wave_vector

                                   
class SineWaveDataset82(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.num_timesteps = 300
        self.next_vector = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        amplitude = self.data.iloc[idx, 0]
        period = self.data.iloc[idx, 1]
        sine_wave = eval(self.data.iloc[idx, 2])  # assuming the sine_wave is stored as a string representation of a list
        
        # L1: Explicit Time Resetting with period 
        amplitude_vector = torch.tensor([amplitude] * self.num_timesteps, dtype=torch.float32)
        time_vector = SineWaveDataset81.next_vector
        # Use the next_vector from SineWaveDataset81 if it's the first call
        self.next_vector = add_noise(time_vector - SineWaveDataset81.decrease_vector)
        period_vector = torch.tensor([period] * self.num_timesteps, dtype=torch.float32)
        sine_wave_vector = torch.tensor(sine_wave, dtype=torch.float32)
        input_vector = torch.stack([amplitude_vector, period_vector, time_vector], dim=1)  
        return input_vector, sine_wave_vector
                                   
                                   
class SineWaveDataset83(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.num_timesteps = 300
        self.next_vector = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        amplitude = self.data.iloc[idx, 0]
        period = self.data.iloc[idx, 1]
        sine_wave = eval(self.data.iloc[idx, 2])  # assuming the sine_wave is stored as a string representation of a list
        
        # L1: Explicit Time Resetting with period 
        amplitude_vector = torch.tensor([amplitude] * self.num_timesteps, dtype=torch.float32)
        time_vector = SineWaveDataset82.next_vector
        # Use the next_vector from SineWaveDataset81 if it's the first call
        self.next_vector = add_noise(time_vector - SineWaveDataset81.decrease_vector)
        period_vector = torch.tensor([period] * self.num_timesteps, dtype=torch.float32)
        sine_wave_vector = torch.tensor(sine_wave, dtype=torch.float32)
        input_vector = torch.stack([amplitude_vector, period_vector, time_vector], dim=1)  
        return input_vector, sine_wave_vector
                                   
                                
class SineWaveDataset84(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.num_timesteps = 300
        self.next_vector = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        amplitude = self.data.iloc[idx, 0]
        period = self.data.iloc[idx, 1]
        sine_wave = eval(self.data.iloc[idx, 2])  # assuming the sine_wave is stored as a string representation of a list
        
        # L1: Explicit Time Resetting with period 
        amplitude_vector = torch.tensor([amplitude] * self.num_timesteps, dtype=torch.float32)
        time_vector = SineWaveDataset83.next_vector

        # Use the next_vector from SineWaveDataset81 if it's the first call
        self.next_vector = add_noise(time_vector - SineWaveDataset81.decrease_vector)

        period_vector = torch.tensor([period] * self.num_timesteps, dtype=torch.float32)
        sine_wave_vector = torch.tensor(sine_wave, dtype=torch.float32)
        input_vector = torch.stack([amplitude_vector, period_vector, time_vector], dim=1)  
        # Shape: [num_timesteps, 2]
        return input_vector, sine_wave_vector
                                   
                                   
                                   
                                   
class SineWaveDataset85(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.num_timesteps = 300
        self.next_vector = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        amplitude = self.data.iloc[idx, 0]
        period = self.data.iloc[idx, 1]
        sine_wave = eval(self.data.iloc[idx, 2])  # assuming the sine_wave is stored as a string representation of a list
        
        # L1: Explicit Time Resetting with period 
        amplitude_vector = torch.tensor([amplitude] * self.num_timesteps, dtype=torch.float32)
        time_vector = SineWaveDataset84.next_vector

        # Use the next_vector from SineWaveDataset81 if it's the first call
        # if self.next_vector is None:
        #     self.next_vector = add_noise(time_vector - SineWaveDataset81.decrease_vector)

        period_vector = torch.tensor([period] * self.num_timesteps, dtype=torch.float32)
        sine_wave_vector = torch.tensor(sine_wave, dtype=torch.float32)
        input_vector = torch.stack([amplitude_vector, period_vector, time_vector], dim=1)  
        # Shape: [num_timesteps, 2]
        return input_vector, sine_wave_vector
                                                                  