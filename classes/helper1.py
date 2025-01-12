import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
from snntorch import spikeplot as splt
import matplotlib.pyplot as plt
from classes.ATanSurrogate1 import ATanSurrogate1


'''all helper functions that we need'''

#creates a weight matrix given sparsity with log-normal distribution of values
def conn_mx(rows, columns, sparsity):
    # Calculate the number of non-zero entries based on sparseness
    num_non_zero_entries = int(rows * columns * sparsity)

    # Initialize the matrix with zeros
    conn_mx = torch.zeros(rows, columns)

    # Randomly select indices to set to the specified value
    indices = torch.randperm(rows * columns)[:num_non_zero_entries]

    # Initialize non-zero values using log normal distribution
    mu = -0.64
    sigma = 0.51
    log_normal_values = torch.empty(indices.shape).normal_(mean=mu, std=sigma).exp_()
    conn_mx.view(-1)[indices] = log_normal_values

    return conn_mx


# creates an excitatory and inhibitory matrix with sparsity and lognormal distribution hardcoded
def hid_mx(rows, columns, num_excitatory, num_inhibitory):
    # hard coded sparsity

    # Initialize the weight matrix
    weight_matrix = np.zeros((num_excitatory + num_inhibitory, num_excitatory + num_inhibitory))

    # Set excitatory to excitatory connections
    weight_matrix[:num_excitatory, :num_excitatory] = np.random.choice([0, 1], size=(num_excitatory, num_excitatory), p=[1-0.16, 0.16])

    # Set excitatory to inhibitory connections
    weight_matrix[:num_excitatory, num_excitatory:] = np.random.choice([0, 1], size=(num_excitatory, num_inhibitory), p=[1-0.205, 0.205])

    # Set inhibitory to excitatory connections
    weight_matrix[num_excitatory:, :num_excitatory] = np.random.choice([0, -1], size=(num_inhibitory, num_excitatory),p=[1-0.252, 0.252])

    # Set inhibitory to inhibitory connections
    weight_matrix[num_excitatory:, num_excitatory:] = np.random.choice([0, -1], size=(num_inhibitory, num_inhibitory), p=[1-0.284, 0.284] )

    # Initialize non-zero values using log normal distribution
    mu = -0.64
    sigma = 0.51
    non_zero_indices = np.where(weight_matrix != 0)
    weight_matrix[non_zero_indices] = np.random.lognormal(mean=mu, sigma=sigma, size=non_zero_indices[0].shape)

    # Multiply the last num_inhibitory rows by -10
    weight_matrix[-num_inhibitory:, :] *= -10

    return torch.tensor(weight_matrix.astype(np.float32))

# creates an excitatory and inhibitory matrix
def hid_mx3I(num_excitatory, num_inhibitory, num_iPV, num_iSST, num_iHtr, p_nn):

    # Why are there so many neurons :( 

    # Initialize the weight matrix
    weight_matrix = np.zeros((num_excitatory + num_inhibitory, num_excitatory + num_inhibitory))

    # Excitatory connections

    # excitatory to excitatory
    weight_matrix[:num_excitatory, :num_excitatory] = np.random.choice([0, 1], size=(num_excitatory, num_excitatory), p=[1-p_nn['e_e'], p_nn['e_e']])
    # excitatory to inhibitory PV
    weight_matrix[:num_excitatory, num_excitatory:num_excitatory+num_iPV] = np.random.choice([0, 1], size=(num_excitatory, num_iPV), p=[1-p_nn['e_PV'], p_nn['e_PV']])
    # excitatory to inhibitory SST
    weight_matrix[:num_excitatory, num_excitatory+num_iPV:num_excitatory+num_iPV+num_iSST] = np.random.choice([0, 1], size=(num_excitatory, num_iSST), p=[1-p_nn['e_SST'], p_nn['e_SST']])
    # excitatory to inhibitory Htr
    weight_matrix[:num_excitatory, num_excitatory+num_iPV+num_iSST:] = np.random.choice([0, 1], size=(num_excitatory, num_iHtr), p=[1-p_nn['e_Htr'], p_nn['e_Htr']])


    # Inhibitory connections

    # inhibitory PV to excitatory
    weight_matrix[num_excitatory:num_excitatory+num_iPV, :num_excitatory] = np.random.choice([0, -1], size=(num_iPV, num_excitatory), p=[1-p_nn['PV_e'], p_nn['PV_e']])
    # inhibitory PV to inhibitory PV
    weight_matrix[num_excitatory:num_excitatory+num_iPV, num_excitatory:num_excitatory+num_iPV] = np.random.choice([0, -1], size=(num_iPV, num_iPV), p=[1-p_nn['PV_PV'], p_nn['PV_PV']])
    # inhibitory PV to inhibitory Htr
    weight_matrix[num_excitatory:num_excitatory+num_iPV, num_excitatory+num_iPV:num_excitatory+num_iPV+num_iSST] = np.random.choice([0, -1], size=(num_iPV, num_iSST), p=[1-p_nn['PV_SST'], p_nn['PV_SST']])
    # inhibitory PV to inhibitory SST
    weight_matrix[num_excitatory:num_excitatory+num_iPV, num_excitatory+num_iPV+num_iSST:] = np.random.choice([0, -1], size=(num_iPV, num_iHtr), p=[1-p_nn['PV_Htr'], p_nn['PV_Htr']]) 

    # inhibitory SST to excitatory
    weight_matrix[num_excitatory+num_iPV:num_excitatory+num_iPV+num_iSST, :num_excitatory] = np.random.choice([0, -1], size=(num_iSST, num_excitatory), p=[1-p_nn['SST_e'], p_nn['SST_e']])
    # inhibitory SST to inhibitory PV
    weight_matrix[num_excitatory+num_iPV:num_excitatory+num_iPV+num_iSST, num_excitatory:num_excitatory+num_iPV] = np.random.choice([0, -1], size=(num_iSST, num_iPV), p=[1-p_nn['SST_PV'], p_nn['SST_PV']])
    # inhibitory SST to inhibitory Htr
    weight_matrix[num_excitatory+num_iPV:num_excitatory+num_iPV+num_iSST, num_excitatory+num_iPV:num_excitatory+num_iPV+num_iSST] = np.random.choice([0, -1], size=(num_iSST, num_iSST), p=[1-p_nn['SST_SST'], p_nn['SST_SST']])
    # inhibitory SST to inhibitory SST
    weight_matrix[num_excitatory+num_iPV:num_excitatory+num_iPV+num_iSST, num_excitatory+num_iPV+num_iSST:] = np.random.choice([0, -1], size=(num_iSST, num_iHtr), p=[1-p_nn['SST_Htr'], p_nn['SST_Htr']]) 

    # inhibitory SST to excitatory
    weight_matrix[num_excitatory+num_iPV+num_iSST:, :num_excitatory] = np.random.choice([0, -1], size=(num_iHtr, num_excitatory), p=[1-p_nn['Htr_e'], p_nn['Htr_e']])
    # inhibitory SST to inhibitory PV
    weight_matrix[num_excitatory+num_iPV+num_iSST:, num_excitatory:num_excitatory+num_iPV] = np.random.choice([0, -1], size=(num_iHtr, num_iPV), p=[1-p_nn['Htr_PV'], p_nn['Htr_PV']])
    # inhibitory SST to inhibitory Htr
    weight_matrix[num_excitatory+num_iPV+num_iSST:, num_excitatory+num_iPV:num_excitatory+num_iPV+num_iSST] = np.random.choice([0, -1], size=(num_iHtr, num_iSST), p=[1-p_nn['Htr_SST'], p_nn['Htr_SST']])
    # inhibitory SST to inhibitory SST
    weight_matrix[num_excitatory+num_iPV+num_iSST:, num_excitatory+num_iPV+num_iSST:] = np.random.choice([0, -1], size=(num_iHtr, num_iHtr), p=[1-p_nn['Htr_Htr'], p_nn['Htr_Htr']]) 


    # Initialize non-zero values using log normal distribution
    mu = -0.64
    sigma = 0.51
    non_zero_indices = np.where(weight_matrix != 0)
    weight_matrix[non_zero_indices] = np.random.lognormal(mean=mu, sigma=sigma, size=non_zero_indices[0].shape)

    # Multiply the last num_inhibitory rows by -10
    weight_matrix[-num_inhibitory:, :] *= -10

    return torch.tensor(weight_matrix.astype(np.float32))

#create hid_mx with only PV as interneuron
def hid_mx_iPV(num_excitatory, num_iPV, p_nn):

    # Initialize the weight matrix
    weight_matrix = np.zeros((num_excitatory + num_iPV, num_excitatory + num_iPV))

    # Excitatory connections

    weight_matrix[:num_excitatory, :num_excitatory] = np.random.choice([0, 1], size=(num_excitatory, num_excitatory), p=[1-p_nn['e_e'], p_nn['e_e']])
    weight_matrix[:num_excitatory, num_excitatory:num_excitatory+num_iPV] = np.random.choice([0, 1], size=(num_excitatory, num_iPV), p=[1-p_nn['e_PV'], p_nn['e_PV']])

    # inhibitory PV connections
    
    weight_matrix[num_excitatory:num_excitatory+num_iPV, :num_excitatory] = np.random.choice([0, -1], size=(num_iPV, num_excitatory), p=[1-p_nn['PV_e'], p_nn['PV_e']])
    weight_matrix[num_excitatory:num_excitatory+num_iPV, num_excitatory:num_excitatory+num_iPV] = np.random.choice([0, -1], size=(num_iPV, num_iPV), p=[1-p_nn['PV_PV'], p_nn['PV_PV']])

    # Initialize non-zero values using log normal distribution
    mu = -0.64
    sigma = 0.51
    non_zero_indices = np.where(weight_matrix != 0)
    weight_matrix[non_zero_indices] = np.random.lognormal(mean=mu, sigma=sigma, size=non_zero_indices[0].shape)

    # Multiply the last num_inhibitory rows by -10
    weight_matrix[-num_iPV:, :] *= -10

    return torch.tensor(weight_matrix.astype(np.float32))



#create hid_mx with only SST as interneuron
def hid_mx_iSST(num_excitatory, num_iSST, p_nn):

    # Initialize the weight matrix
    weight_matrix = np.zeros((num_excitatory + num_iSST, num_excitatory + num_iSST))

    # Excitatory connections

    weight_matrix[:num_excitatory, :num_excitatory] = np.random.choice([0, 1], size=(num_excitatory, num_excitatory), p=[1-p_nn['e_e'], p_nn['e_e']])
    weight_matrix[:num_excitatory, num_excitatory:num_excitatory+num_iSST] = np.random.choice([0, 1], size=(num_excitatory, num_iSST), p=[1-p_nn['e_SST'], p_nn['e_SST']])

    # inhibitory SST connections
    
    weight_matrix[num_excitatory:num_excitatory+num_iSST, :num_excitatory] = np.random.choice([0, -1], size=(num_iSST, num_excitatory), p=[1-p_nn['SST_e'], p_nn['SST_e']])
    weight_matrix[num_excitatory:num_excitatory+num_iSST, num_excitatory:num_excitatory+num_iSST] = np.random.choice([0, -1], size=(num_iSST, num_iSST), p=[1-p_nn['SST_SST'], p_nn['SST_SST']])

    # Initialize non-zero values using log normal distribution
    mu = -0.64
    sigma = 0.51
    non_zero_indices = np.where(weight_matrix != 0)
    weight_matrix[non_zero_indices] = np.random.lognormal(mean=mu, sigma=sigma, size=non_zero_indices[0].shape)

    # Multiply the last num_inhibitory rows by -10
    weight_matrix[-num_iSST:, :] *= -10

    return torch.tensor(weight_matrix.astype(np.float32))


#create hid_mx with only SST as interneuron
def hid_mx_iPS(num_excitatory, num_iPV, num_iSST, p_nn):

    # Initialize the weight matrix
    weight_matrix = np.zeros((num_excitatory + num_iPV+ num_iSST, num_excitatory + num_iPV + num_iSST))

    # Excitatory connections
    
    weight_matrix[:num_excitatory, :num_excitatory] = np.random.choice([0, 1], size=(num_excitatory, num_excitatory), p=[1-p_nn['e_e'], p_nn['e_e']])
    weight_matrix[:num_excitatory, num_excitatory:num_excitatory+num_iPV] = np.random.choice([0, 1], size=(num_excitatory, num_iPV), p=[1-p_nn['e_PV'], p_nn['e_PV']])
    weight_matrix[:num_excitatory, num_excitatory+num_iPV:num_excitatory+num_iPV+num_iSST] = np.random.choice([0, 1], size=(num_excitatory, num_iSST), p=[1-p_nn['e_SST'], p_nn['e_SST']])

    # inhibitory PV connections
    
    weight_matrix[num_excitatory:num_excitatory+num_iPV, :num_excitatory] = np.random.choice([0, -1], size=(num_iPV, num_excitatory), p=[1-p_nn['PV_e'], p_nn['PV_e']])
    weight_matrix[num_excitatory:num_excitatory+num_iPV, num_excitatory:num_excitatory+num_iPV] = np.random.choice([0, -1], size=(num_iPV, num_iPV), p=[1-p_nn['PV_PV'], p_nn['PV_PV']])
    weight_matrix[num_excitatory:num_excitatory+num_iPV, num_excitatory+num_iPV: num_excitatory+num_iPV+num_iSST] = np.random.choice([0, -1], size=(num_iPV, num_iSST), p=[1-p_nn['PV_SST'], p_nn['PV_SST']])
    
    
    # inhibitory SST connections
    
    weight_matrix[num_excitatory+num_iPV : num_excitatory+num_iPV+num_iSST, :num_excitatory] = np.random.choice([0, -1], size=(num_iSST, num_excitatory), p=[1-p_nn['SST_e'], p_nn['SST_e']])
    weight_matrix[num_excitatory+num_iPV:num_excitatory+num_iPV+num_iSST, num_excitatory:num_excitatory+num_iPV] = np.random.choice([0, -1], size=(num_iSST, num_iPV), p=[1-p_nn['SST_PV'], p_nn['SST_PV']])
    weight_matrix[num_excitatory+num_iPV:num_excitatory+num_iPV+num_iSST , num_excitatory+num_iPV : num_excitatory+num_iPV+num_iSST] = np.random.choice([0, -1], size=(num_iSST, num_iSST), p=[1-p_nn['SST_SST'], p_nn['SST_SST']])

    # Initialize non-zero values using log normal distribution
    mu = -0.64
    sigma = 0.51
    non_zero_indices = np.where(weight_matrix != 0)
    weight_matrix[non_zero_indices] = np.random.lognormal(mean=mu, sigma=sigma, size=non_zero_indices[0].shape)

    # Multiply the last num_inhibitory rows by -10
    weight_matrix[-(num_iPV+num_iSST):, :] *= -10

    return torch.tensor(weight_matrix.astype(np.float32))




#plots spike rasters given spike array(time x neuron)
def plot_spike_tensor(spk_tensor, title):
    # Generate the plot
    spk_tensor = spk_tensor.T
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot spikes
    splt.raster(spk_tensor, ax, s=0.5, c="black")  # Transpose to align with neurons on y-axis

    # Set labels and title
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Neuron")
    ax.set_title(title)

    plt.show()

#calculate criticality given spike recording and (size of each time bin)
def simple_branching_param(bin_size, spikes):  # spikes in shape of [units, time]
    run_time = spikes.shape[1]
    nbins = spikes.shape[1]
    # nbins = int(np.round(run_time / bin_size))

    # for every pair of timesteps, determine the number of ancestors
    # and the number of descendants
    numA = torch.zeros(nbins - 1)
    # number of ancestors for each bin
    numD = torch.zeros(nbins - 1)
    # number of descendants for each ancestral bin
    i = 0
    while i < (numA.size(0) - 1):
        numA[i] = torch.sum(spikes[:, i] == 1).item()
        numD[i] = torch.sum(spikes[:, i + bin_size] == 1).item()

        # Check if numA[i] is 0, and remove numA[i] and numD[i] if it is
        if numA[i] == 0:
            numA = torch.cat((numA[:i], numA[i+1:]))
            numD = torch.cat((numD[:i], numD[i+1:]))
        else:
            i+=1

    # the ratio of descendants per ancestor
    d = numD / numA
    bscore = torch.nanmean(d)
    return bscore

# Synchrony -- Fano Factor
def fano_factor(seq_len, spike):
    # Calculate value similar to the Fano factor to estimate synchrony quickly
    # During each bin, calculate the variance of the number of spikes per neuron divided by the mean of the number of spikes per neuron
    # The Fano factor during one interval is equal to the mean of the values calculated for each bin in it
    # Spike should have dims of neuron, time
    # Returned fano factor should have dims of trial
    len_bins = 10  # ms
    n_bins = int(round(seq_len / len_bins))
    fano_all = torch.zeros(n_bins)
    for i in range(n_bins):
        spike_slice = spike[:, i * len_bins:(i + 1) * len_bins]
        spikes_per_neuron = torch.sum(spike_slice, axis=1)
        variance_spikes = torch.var(spikes_per_neuron)
        mean_spikes = torch.mean(spikes_per_neuron)
        fano_bin = variance_spikes / mean_spikes if mean_spikes != 0 else 0
        fano_all[i] = fano_bin
    n_fano = torch.mean(fano_all)
    return n_fano

#function calculating arctangent that is used in SpikingNeuron1.py
def atan1(alpha=2.0):
    """ArcTan surrogate gradient enclosed with a parameterized slope."""
    return ATanSurrogate1(alpha)

#counts total number of spikes
def count_spikes(tensor):
    return torch.count_nonzero(tensor).item()

