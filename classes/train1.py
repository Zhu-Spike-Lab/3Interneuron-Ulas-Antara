import torch
from classes.helper1 import simple_branching_param, fano_factor, count_spikes, sparsity


#function to train and save the variables to npz files!
def train_model16(args):
    job, model, optimizer, dataloader,criterion, taskid, ineuron, num_epochs, num_timesteps= args
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0

        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs, targets

            optimizer.zero_grad()

            outputs = torch.empty(0, dtype=torch.float32, requires_grad=True)
            spikes = torch.empty(0, dtype=torch.float32)  # Initialize spike storage

            for input, _ in zip(inputs, targets):
                output, spike = model(input)
                spike = spike.T

                # file outputs and spikes wil lhave an extra dimension of 25
                outputs = torch.cat((outputs, output.view(1, -1)))
                spikes = torch.cat((spikes, spike.unsqueeze(0)))
                    
            loss = criterion(outputs, targets)

            print("loss:", loss)
            loss.backward()
            optimizer.step()
            
            # try
            model.positive_negative_weights()
            
            epoch_loss += loss.item()

            if epoch % 10 == 0:
                torch.save({
                    "task_loss":criterion.task_loss.item(),
                    "spikes":spikes,
                    "input_weights":model.l1.weight.data,
                    "rec_weights":model.rlif1.recurrent.weight.data,
                    "output_weights":model.l2.weight.data,
                    "inputs":inputs,
                    "outputs":outputs,
                    "targets":targets}, 
                    f'data_final/task{taskid}_i{ineuron}_job{job}_epoch{epoch}_batch{i}.pth')

# #function to train and save the variables to npz files!
# def train_model(args):
#     model, optimizer, dataloader,step_dataloader, criterion, criterium_idx, num_epochs, num_timesteps= args
#     model.train()
#     for epoch in range(num_epochs):
#         epoch_loss = 0

#         for i, (inputs, targets) in enumerate(dataloader):
#             inputs, targets = inputs, targets

#             optimizer.zero_grad()

#             outputs = torch.empty(0, dtype=torch.float32, requires_grad=True)
#             # firing_rate_per_batch = torch.empty(0, dtype=torch.float32, requires_grad=True)
#             # criticality_per_batch = torch.empty(0, dtype=torch.float32, requires_grad=True)
#             # synchrony_per_batch = torch.empty(0, dtype=torch.float32, requires_grad=True)

#             for input, target in zip(inputs, targets):
#                 output, spikes = model(input)
#                 spikes = spikes.T
#                 outputs = torch.cat((outputs, output.view(1, -1)))

# #                 firing_rate = count_spikes(spikes) / 30000
# #                 firing_rate_per_batch = torch.cat((firing_rate_per_batch, firing_rate))

# #                 criticality = simple_branching_param(1, spikes).reshape([1])
# #                 criticality_per_batch = torch.cat((criticality_per_batch, criticality))

# #                 synchrony_fano_factor = fano_factor(num_timesteps, spikes).reshape([1])
# #                 synchrony_per_batch = torch.cat((synchrony_per_batch, synchrony_fano_factor))

#             # loss = criterion(outputs, targets, criticality_per_batch, firing_rate_per_batch, synchrony_per_batch)
            
#             loss = criterion(outputs, targets)
#             print("loss:", loss)
#             loss.backward()
#             optimizer.step()
            
#             #call positive_negative_weights
#             model.positive_negative_weights()

#             epoch_loss += loss.item()

#             if epoch % 5 == 0:
#                 np.savez(f'dataMP/level{step_dataloader}_loss{criterium_idx}_epoch{epoch}_batch{i}.npz',
#                          task_loss=criterion.task_loss.item(),
#                          spikes=spikes.detach().numpy(),
#                          input_weights=model.l1.weight.data.detach().numpy(),
#                          rec_weights=model.rlif1.recurrent.weight.data.detach().numpy(),
#                          output_weights=model.l2.weight.data.detach().numpy(),
#                          inputs=inputs.detach().numpy(),
#                          outputs=outputs.detach().numpy(),
#                          targets=targets.detach().numpy())

                
# #function to train and save the variables to npz files!
# #model8 traines on different dataloaders each 200th epoch
# def train_model8(args):
#     model, optimizer, dataloader_list,step_dataloader, criterion, criterium_idx, num_epochs, num_timesteps= args
#     model.train()
#     for epoch in range(num_epochs):
#         epoch_loss = 0

#         if epoch < 200:
#             dataloader = dataloader_list[0]
#             print(1)
#         elif epoch < 400:
#             dataloader = dataloader_list[1]
#             print(2)
#         elif epoch < 600:
#             dataloader = dataloader_list[2]
#             print(3)
#         elif epoch < 800:
#             dataloader = dataloader_list[3]
#             print(4)
#         elif epoch < 1000:
#             dataloader = dataloader_list[4]
#             print(5)
            
#         for i, (inputs, targets) in enumerate(dataloader):
#             inputs, targets = inputs, targets

#             optimizer.zero_grad()

#             outputs = torch.empty(0, dtype=torch.float32, requires_grad=True)
#             # firing_rate_per_batch = torch.empty(0, dtype=torch.float32, requires_grad=True)
#             # criticality_per_batch = torch.empty(0, dtype=torch.float32, requires_grad=True)
#             # synchrony_per_batch = torch.empty(0, dtype=torch.float32, requires_grad=True)

#             for input, target in zip(inputs, targets):
#                 output, spikes = model(input)
#                 spikes = spikes.T
#                 outputs = torch.cat((outputs, output.view(1, -1)))

# #                 firing_rate = count_spikes(spikes) / 30000
# #                 firing_rate_per_batch = torch.cat((firing_rate_per_batch, firing_rate))

# #                 criticality = simple_branching_param(1, spikes).reshape([1])
# #                 criticality_per_batch = torch.cat((criticality_per_batch, criticality))

# #                 synchrony_fano_factor = fano_factor(num_timesteps, spikes).reshape([1])
# #                 synchrony_per_batch = torch.cat((synchrony_per_batch, synchrony_fano_factor))

#             # loss = criterion(outputs, targets, criticality_per_batch, firing_rate_per_batch, synchrony_per_batch)
            
#             loss = criterion(outputs, targets)
#             print("loss:", loss)
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()

#             if epoch % 5 == 0:
#                 np.savez(f'dataMP/level{step_dataloader}_loss{criterium_idx}_epoch{epoch}_batch{i}.npz',
#                          task_loss=criterion.task_loss.item(),
#                          spikes=spikes.detach().numpy(),
#                          input_weights=model.l1.weight.data.detach().numpy(),
#                          rec_weights=model.rlif1.recurrent.weight.data.detach().numpy(),
#                          output_weights=model.l2.weight.data.detach().numpy(),
#                          inputs=inputs.detach().numpy(),
#                          outputs=outputs.detach().numpy(),
#                          targets=targets.detach().numpy())
