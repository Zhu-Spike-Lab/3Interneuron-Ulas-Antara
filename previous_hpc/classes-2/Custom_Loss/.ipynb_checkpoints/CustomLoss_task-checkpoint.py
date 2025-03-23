import torch.nn as nn
import torch

#includes task loss
class CustomLoss_task(nn.Module):
    def __init__(self,batch_size=25):
        super(CustomLoss_task, self).__init__()

    def forward(self, outputs, targets):
        task_loss = nn.MSELoss()(outputs.squeeze(), targets)
        self.task_loss = task_loss

        return task_loss
