import flappy_bird_gymnasium
import gymnasium
import numpy as np
from dqn import DQN
from experience import ReplayMemory
import itertools
import yaml
import random
import torch
import os
import matplotlib
from torch import nn
from datetime import datetime,timedelta
import argparse
import matplotlib.pyplot as plt
DATE_FORMAT = "%Y-%m-%d-%H-%M-%S"
RUNS_DIR="runs"
os.makedirs(RUNS_DIR,exist_ok=True)
matplotlib.use("Agg")
class Agent():
    def __init__(self,hyperparameters_set):
        with open('hyperparameter.yml','r') as file:
            all_hyperparameters = yaml.safe_load(file)
            hyperparameters = all_hyperparameters[hyperparameters_set]
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.stop_on_reward=hyperparameters['stop_on_reward']
        self.fc1_nodes=hyperparameters['fc1_nodes']
        self.env_make_params=hyperparameters.get('env_make_params',{})
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer=None
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g=hyperparameters['discount_factor_g']
        self.LOG_FILE=os.path.join(RUNS_DIR, f'{hyperparameters_set}.log')
        self.MODEL_FILE=os.path.join(RUNS_DIR, f'{hyperparameters_set}.pt')
        self.GRAPH_FILE=os.path.join(RUNS_DIR, f'{hyperparameters_set}.png')
        self.env_id=hyperparameters['env_id']
        self.enable_double_dqn=hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn=hyperparameters['enable_dueling_dqn']

    def run(self,istraining=True,render=False):
        env = gymnasium.make(self.env_id, render_mode="human" if render else None,**self.env_make_params)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        reward_memory = []
        epsilon_history = []
        policy_dqn=DQN(num_states,num_actions,self.fc1_nodes,self.enable_dueling_dqn)
        replay_memory=[]
        epsilon=0.0
        if istraining:
            replay_memory = ReplayMemory(self.replay_memory_size)
            epsilon=self.epsilon_init
            target_dqn=DQN(num_states,num_actions,self.fc1_nodes,self.enable_dueling_dqn)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            step_count=0
            self.optimizer= torch.optim.Adam(policy_dqn.parameters(),lr=self.learning_rate_a)
            epsilon_history=[]
            best_reward = -9999999
            last_graph_update_time = datetime.now()
            for episode in itertools.count():
                state, _ = env.reset()
                state = torch.tensor(state,dtype=torch.float,device="cpu")
                terminated = False
                episode_reward = 0.0
                while not terminated and episode_reward<self.stop_on_reward:
                    # Next action:
                    # (feed the observation to your agent here)
                    if istraining and random.random()<epsilon:
                        action = env.action_space.sample()
                    else:
                        with torch.no_grad():
                            action=policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax().item()

                    # Processing:
                    new_state, reward, terminated, _, info = env.step(action)
                    episode_reward += reward
                    new_state = torch.tensor(new_state,dtype=torch.float,device="cpu")
                    reward = torch.tensor(reward,dtype=torch.float,device="cpu")
                    action_tensor=torch.tensor(action,dtype=torch.long)
                    replay_memory.append((state,action_tensor,new_state,reward,terminated))
                    step_count=step_count+1
                    state=new_state
                    # Checking if the player is still alive
                    if len(replay_memory)>self.mini_batch_size:
                        mini_batch=replay_memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch,policy_dqn,target_dqn)
                        if step_count>self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            step_count=0
                reward_memory.append(episode_reward)
                if episode_reward > best_reward:
                    log_message=f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f}({(episode_reward-best_reward)/episode_reward*100})"
                    print(log_message)
                    with open(self.LOG_FILE,'a') as f:
                        f.write(log_message+'\n')
                    torch.save(policy_dqn.state_dict(),self.MODEL_FILE)
                    best_reward = episode_reward
                current_time=datetime.now()
                if current_time-last_graph_update_time>timedelta(seconds=10):
                    self.save_graph(reward_memory,epsilon_history)
                    last_graph_update_time=current_time
                epsilon=max(epsilon*self.epsilon_decay,self.epsilon_min)
                epsilon_history.append(epsilon)
                print(f"Episode {episode} finished")
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()
            state,_=env.reset()
            state=torch.tensor(state,dtype=torch.float,device="cpu")
            terminated=False
            episode_reward = 0.0
            while not terminated and episode_reward<self.stop_on_reward:
                with torch.no_grad():
                    action=policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax().item()
                new_state, reward, terminated,_,info = env.step(action)
                episode_reward += reward
                state=torch.tensor(new_state,dtype=torch.float,device='cpu')
            env.close()
    def save_graph(self,rewards_per_episode,epsilon_history):
        fig=plt.figure(1)
        plt.clf()
        mean_rewards=np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x]=np.mean(rewards_per_episode[max(0,x-99):(x+1)])
        plt.subplot(121)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.plot(mean_rewards)
        plt.subplot(122)
        plt.ylabel('Epsilon_Decay')
        plt.plot(epsilon_history)
        plt.subplots_adjust(wspace=1.0,hspace=1.0)
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)
    def optimize(self,mini_batch,policy_dqn,target_dqn):
        states,actions,new_states,rewards,terminations=zip(*mini_batch)
        states=torch.stack(states)
        actions=torch.stack(actions)
        new_states=torch.stack(new_states)
        rewards=torch.stack(rewards)
        terminations=torch.tensor(terminations).float()
        with torch.no_grad():
            if self.enable_double_dqn:
                best_action_from_policy=policy_dqn(new_states).argmax(dim=1)
                target_q=rewards+(1-terminations)*self.discount_factor_g*target_dqn(new_states).gather(dim=1,index=best_action_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q=rewards+(1-terminations)*self.discount_factor_g*target_dqn(new_states).max(dim=1)[0]
        current_q=policy_dqn(states).gather(dim=1,index=actions.unsqueeze(dim=1).long()).squeeze()
        loss=self.loss_fn(current_q,target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Train or test model')
    parser.add_argument('hyperparameters',help='')
    parser.add_argument('--train',help='Training mode',action='store_true')
    args=parser.parse_args()
    dq1=Agent(hyperparameters_set=args.hyperparameters)
    if args.train:
        dq1.run(istraining=True)
    else:
        dq1.run(istraining=False,render=True)