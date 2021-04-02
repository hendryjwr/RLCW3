# https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py
import torch
from torch import nn
import numpy as np
from collections import deque
import random

import gym 

env = gym.make('Assault-ram-v0')
env.reset()
env.render()

# get_action_meanings() = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']  (6 possible actions for this game)
# step() = (array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  96, 254,
    #      0,   0,   0,   6, 100, 100, 100,   0,  54,   0,   0,   0, 253,
    #      0,   0, 192,   0, 136, 252,   2,  34, 226,  34, 162,  83, 162,
    #      6, 188, 255,   0,  25,   0, 253,   0, 253, 128,  64, 128, 128,
    #     64, 128,   0,   0,   0,   0,   0,   0,  30,  30,  17,  17,   0,
    #    253,   0, 127,  83,  68,  64,  19,  24,   0, 253,   0,   0,   0,
    #      0,   0,  34, 162,   0, 254,   0, 254,   0, 254,   0, 254,   0,
    #    254, 144,  60,   0,   0,   0,   0,   0,  80, 254,   4, 218,  69,
    #      0,  10,   0,   5,   0,   0, 255, 248,   0,   0,  64,   0, 172,
    #      0,   0,   0, 248, 251, 189, 251,  64, 251,   0, 245], dtype=uint8), 0.0, False, {'ale.lives': 4})
# step returns ob (ram in this case), reward, if game is over (Bool) and a dict of lives remaining


for i in range(1):
    print(env.step(1))
    env.render()   
print('env is: ', env)


class Agent(Agent):

    def __init__(self, state_dim, action_dim, save_dir): # Add save_dir if implemented (idk what this is atm)
        super().__init__(state_dim, action_dim, save_dir)
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        # self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.epsilon = 1
        self.epsilon_rate_decay = 0.99999975
        self.epsilon_min_value = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Mario Net

    def action(self, state):
        """Given a state, what action should be taken. Epsilon-greedy. Outputs an action to be performed, int. """
        
        # Explore
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action)

        # Exploit
        else:
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            # get the action_values from the neural net by passing in the state. (gets back Q values for us to argmax)
            # action_values = self.net(state, model="online") # The net here is the neural network (duh)
            # action = torch.argmax(action_values, axis=1).item()
            action = None

            self.epsilon *= self.epsilon_rate_decay
            self.epsilon = max(self.epsilon_min_value, self.epsilon)

            self.curr_step += 1
            return action

    def memory_storage(self, state, next_state, action, reward, done):
        """Adds recent experience to memory (S, a) and what S' and r was observed. Keeping something semi tabular for an accurate lookup. Replay Buffer"""
        
        if self.use_cuda():
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done))

    def memory_recall(self):
        """Picks a batch of experiences from memory"""
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch)) # Not 100% sure what exactly how this map function works
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze() # Look into why we need to squeeze this

    def learn(self):
        """Update Q values with mini-batch"""
        pass

