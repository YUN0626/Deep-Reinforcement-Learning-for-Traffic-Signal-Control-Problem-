# 參考 https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import sumo_envirment
import  gym
import numpy as np 
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
env = sumo_envirment.Sim(sumo_config="one_intersection_left/one_intersection.sumocfg",GUI=False)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-3
n_actions = env.n_action
# Get the number of state observations
n_observations = env.n_observation
policy_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)



steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randint(0,1)]], dtype=torch.long)
episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)),dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
num_episodes = 2000
q = []
traveltime = []
waitingtime = []
import time
start_time = time.time()
for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    end_time = time.time()
    print('episode: {e} , used time : {t}'.format(e = i_episode,t =end_time-start_time))
    env.reset()
    state = env.runSim()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    epoch_reward = []
    
    for t in count():
        #print(state)
        action = select_action(state)
        action_use = action.item()
        observation, reward,terminated= env.step(action_use)
        
        #print("state{},reward:{}".format(observation,reward))
        epoch_reward.append(-reward)
        reward = torch.tensor([reward])
        
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        
        memory.push(state, action, next_state, reward)

        
        state = next_state

        
        optimize_model()

        
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if i_episode % 250 == 0 :
            np.save("left/output/1225/q_batch32_{}iter_lr3_360".format(i_episode),q)
            np.save("left/output/1225/traveltime_batch32_{}iter_lr3_360".format(i_episode),traveltime)
            np.save("left/output/1225/waitingtime_batch32_{}iter_lr3_360".format(i_episode),waitingtime)
            torch.save(policy_net,'left/output/1225/models/model_{}.pt'.format(i_episode))
        if terminated  :
            #print(ql/t)
            temp = 0
            q.append(max(epoch_reward))
            traveltime.append(sum(env.traveltimes)/len(env.traveltimes))
            for i in env.waiting_times:
                temp = temp + env.waiting_times[i]
            waitingtime.append(temp/len(env.traveltimes))    
            env.close()
            print("action num :{}".format(t))
            break

torch.save(policy_net,'left/output/1225/models/model_2000.pt')
q = np.array(q)
traveltime = np.array(traveltime)
waitingtime = np.array(waitingtime)
np.save("left/output/1225/q_batch32_2000iter_lr3_360",q)
np.save("left/output/1225/traveltime_batch32_2000iter_lr3_360",traveltime)
np.save("left/output/1225/waitingtime_batch32_2000iter_lr3_360",waitingtime)
env = sumo_envirment.Sim(sumo_config="one_intersection_left_test/one_intersection.sumocfg",GUI=False)

state = env.runSim()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
epoch_reward_test = []
print("testing")
for t in count():
    
    action = select_action(state)
    action_use = action.item()
    observation, reward,terminated= env.step(action_use)
    #print("state{},reward:{}".format(observation,reward))
    epoch_reward_test.append(-reward)
    reward = torch.tensor([reward])
    
    if terminated:
        next_state = None
    else:
        next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    state = next_state
    if terminated :
        env.close()
        break
np.save("left/output/1225/test/q_batch32_2000iter_lr3_360",epoch_reward_test)
