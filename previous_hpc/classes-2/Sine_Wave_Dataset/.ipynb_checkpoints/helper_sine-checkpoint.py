import torch

#for sine wave dataset
def add_noise(time_vector, std=3.0, mean=0.0):
    noise = torch.normal(mean, std, size=time_vector.size())
    return time_vector + noise

# def decrease_stepwise(time_vector):
#     decrease_vector = torch.tensor(time_vector/5)
#     return time_vector - decrease_vector