from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

# L500: Changing Amplitude (20-80), Changing Period (40-100), Clock-like Input  Noisy input
class SineWaveDataset500(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.num_timesteps = 300
        #timesteps manually encoded in target_output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        amplitude = self.data.iloc[idx, 0]
        period = self.data.iloc[idx, 1]
        sine_wave = eval(self.data.iloc[idx, 2])  # assuming the sine_wave is stored as a string representation of a list

        # L5: Noisy A,P throughout
        amplitude_vector = torch.normal(mean=amplitude, std=1.0, size=(self.num_timesteps,), dtype=torch.float32)
        period_vector = torch.normal(mean=period, std=1.0, size=(self.num_timesteps,), dtype=torch.float32)
        time_vector = torch.tensor([i for i in range(period)] * (self.num_timesteps//int(period)+1), dtype=torch.float32)
        time_vector = time_vector[:self.num_timesteps]
        
        sine_wave_vector = torch.tensor(sine_wave, dtype=torch.float32)
        input_vector = torch.stack([amplitude_vector,period_vector, time_vector],dim=1)  # Shape: [num_timesteps, 2]
        target_vector = torch.stack([sine_wave_vector],dim=1)
        return input_vector, sine_wave_vector