import torch
import torch.nn as nn
import snntorch as snn
from classes.helper1 import conn_mx, hid_mx3I
from torch import RLIF
import numpy as np


# RSNN class that takes 3 inputs
class RSNN_i3(nn.Module):
    def __init__(self):
        super(RSNN_i3, self).__init__()
        num_inputs = 3
        num_hidden = 200
        num_output = 1
        beta = 0.85
        pe_e = 0.16

        p_nn = {'e_e': 0.16, 'e_PV': 0.395, 'e_SST': 0.182, 'e_Htr': 0.105,
                'PV_e': 0.411, 'PV_PV': 0.451, 'PV_SST': 0.03, 'PV_Htr': 0.22,
                'SST_e': 0.424, 'SST_PV': 0.857, 'SST_SST': 0.082, 'SST_Htr': 0.77,
                'Htr_e': 0.087, 'Htr_PV': 0.02, 'Htr_SST': 0.0625, 'Htr_Htr': 0.028
                } 

        e_beta = 0.85
        iPV_beta = 0.3
        iSST_beta = 0.7
        iHtr_beta = 0.6
    
        # Define the dimensions
        num_excitatory = round(0.8 * num_hidden)
        self.num_excitatory = num_excitatory
        num_inhibitory = num_hidden - num_excitatory

        # Three inhibitory neuron classes: 40% PV, 30% SST, 30% Ht3aR
        num_iPV = round(0.4 * num_inhibitory)
        num_iSST = round(0.3 * num_inhibitory)
        num_iHtr = num_inhibitory - num_iSST - num_iPV

        # Three beta values for E, PV, SST, and Htr3a
        # Values chosen based on spike triggered adaptation behavior of each class
        beta_e = torch.asarray([e_beta] * num_excitatory)
        beta_iPV = torch.asarray([iPV_beta] * num_iPV)   # Little/ no spike frequency adaptation 
        beta_iHtr = torch.asarray([iHtr_beta] * num_iHtr)    # Mostly adapting
        beta_iSST = torch.asarray([iSST_beta] * num_iSST)  # Spike frequency adaptation
        beta = torch.cat((beta_e, beta_iPV, beta_iSST, beta_iHtr)) # create array of betas corresponding to each neuron!

        self.false_neg = []
        self.false_pos = []

        #input to hidden layer
        input_hid_mx = conn_mx(num_inputs, num_hidden, pe_e)
        self.input_hid_mx = input_hid_mx
        self.l1 = nn.Linear(num_inputs,num_hidden)
        self.l1.weight.data = input_hid_mx.T

        # Recurrent layer weight matrix
        hidden_mx = hid_mx3I(num_excitatory, num_inhibitory, num_iPV, num_iSST, num_iHtr, p_nn) 
        self.rlif1 = RLIF1(reset_mechanism="zero",threshold=-55.0, beta=beta, linear_features=num_hidden, all_to_all=True)
        self.rlif1.recurrent.weight.data = hidden_mx.T

        #hidden to output layer
        hid_out_mx = conn_mx(num_hidden,num_output,pe_e)
        self.l2 = nn.Linear(num_hidden, num_output)
        self.l2.weight.data = hid_out_mx.T


    def forward(self, inputs):
        spk1,mem1 = self.rlif1.init_rleaky()
        self.spk1_rec = []
        self.cur2_rec = []

        # print(inputs.shape)
        for step in range(inputs.shape[0]): #300
            cur_input = inputs[step,:]
            cur1 = self.l1(cur_input)
            spk1,mem1 = self.rlif1(cur1, spk1, mem1)
            cur2 = self.l2(spk1)

            self.spk1_rec.append(spk1)
            self.cur2_rec.append(cur2)

        self.spk1_rec = torch.stack(self.spk1_rec)
        self.cur2_rec = torch.stack(self.cur2_rec)
        cur2_rec = self.cur2_rec

        
        
        return self.cur2_rec, self.spk1_rec


    # maintain that weights projected from excitatory neurons remain positive and
    # weights projected from inhibitory neurons remain negative
    # for any weight switching sign, initializes a new weight from normal dist
    def positive_negative_weights(self):

        excitatory_weights = self.rlif1.recurrent.weight.data[:, :self.num_excitatory]
        inhibitory_weights = self.rlif1.recurrent.weight.data[:, self.num_excitatory:]

        #save the number of positives in inhibitory and negatives in excitatory region
        num_false_neg = torch.sum(excitatory_weights < 0).item()
        num_false_pos = torch.sum(inhibitory_weights > 0).item()

        self.false_neg.append(num_false_neg)
        self.false_pos.append(num_false_pos)

        # Clamp switched sign values at 0
        excitatory_weights.clamp_(min=0)
        inhibitory_weights.clamp_(max=0)

        mu = -0.64
        sigma = 0.51


        #change the code so that for any vanishing excitatory neuron, populate another excitatory.

        #sets all zero weights to zero, returns tensor of length # zeros. Each entry is (index row, index column)
        excitatory_zero_indices = (self.rlif1.recurrent.weight.data[:, :self.num_excitatory] == 0).nonzero(as_tuple=True)
        inhibitory_zero_indices = (self.rlif1.recurrent.weight.data[:, self.num_excitatory:] == 0).nonzero(as_tuple=True)

        if (len(excitatory_zero_indices) > num_false_pos):
            #creates num_false_pos integers between 0-excitatory_zero_indices[0]
            #randomly selects row and column indices, (between 0 and len(excitatory_zero_indices[0]) to change with a future randomly generated value.
            #of the shape (num_false_pos,2) where [:,0] lists the row indices of all new excitatory values, and [:,1] the column
            excitatory_sampled_indices = torch.stack([
                    excitatory_zero_indices[0][torch.randint(len(excitatory_zero_indices[0]), (num_false_pos,))],
                    excitatory_zero_indices[1][torch.randint(len(excitatory_zero_indices[1]), (num_false_pos,))]
                ], dim=1)

            # excitatory_sampled_indices is of shape (num_false_pos,2), listing randomly selected indices for columns and rows for updating the matrix
            
            # generating self.excitatory_changes number of lognormal values
            new_excitatory_values = torch.from_numpy(np.random.lognormal(mean=mu, sigma=sigma, size=num_false_pos)).float()
            
            #update the recurrent weight data with new randomly created lognormal values
            self.rlif1.recurrent.weight.data[excitatory_sampled_indices[:, 0], excitatory_sampled_indices[:, 1]] = new_excitatory_values

        if (len(inhibitory_zero_indices) > num_false_neg):
            inhibitory_sampled_indices = torch.stack([
                    inhibitory_zero_indices[0][torch.randint(len(inhibitory_zero_indices[0]), (num_false_neg,))],
                    inhibitory_zero_indices[1][torch.randint(len(inhibitory_zero_indices[1]), (num_false_neg,))]
                ], dim=1)

            new_inhibitory_values = -torch.from_numpy(np.random.lognormal(mean=mu, sigma=sigma, size=num_false_neg)).float()
            self.rlif1.recurrent.weight.data[inhibitory_sampled_indices[:, 0], self.num_excitatory + inhibitory_sampled_indices[:, 1]] = new_inhibitory_values
