import torch
from torch import nn
import torch.nn.functional as F
class DQN(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim,enable_dueling_dqn=True):
        super(DQN,self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,action_dim)
        self.enable_dueling_dqn=enable_dueling_dqn
        if self.enable_dueling_dqn:
            self.fc_value=nn.Linear(hidden_dim,hidden_dim)
            self.value=nn.Linear(hidden_dim,1)
            self.fc_advantages=nn.Linear(hidden_dim,hidden_dim)
            self.advantages=nn.Linear(hidden_dim,action_dim)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        if self.enable_dueling_dqn:
            v=F.relu(self.fc_value(x))
            val=self.value(v)
            a=F.relu(self.fc_advantages(x))
            adv=self.advantages(a)
            q=val+adv-torch.mean(adv,dim=1,keepdim=True)
        else:
            q=self.fc2(x)
        return q
if __name__=='__main__':
    state_dim=12
    action_dim=2
    net=DQN(state_dim,action_dim)
    state=torch.randn(10,state_dim)
    output=net(state)
    print(output)