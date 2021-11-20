import os
import time
import numpy as np
import gym
from gym import spaces
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim

from typing import Tuple


#Environment

class TicTacToe(gym.Env):

    # reward_range = (-np.inf, np.inf)
    # observation_space = spaces.MultiDiscrete([2 for _ in range(0, 9 * 3)])
    # action_space = spaces.Discrete(9)

    """
    Board looks like:
    [0, 1, 2,
     3, 4, 5,
     6, 7, 8]
    """
    winning_streaks = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ]

    def __init__(self, summary: dict = None):
        super().__init__()
        if summary is None:
            summary = {
                "total games": 0,
                "ties": 0,
                "illegal moves": 0,
                "player other wins": 0,
                "player ai wins": 0,
                "random actions" : 0
            }
        self.summary = summary
        self.board = np.zeros(9, dtype="int")
        
    # deterministic
    def seed(self, seed=None):
        pass

    def one_hot_board(self):
        return np.eye(3)[self.board].reshape(-1)
        
    def reset(self):
        # other player = 1, ai_player = -1
        self.board = np.zeros(9, dtype="int")
        return self.one_hot_board()

    def step(self, action, prev_board, player):
        # done values = {0: not over, 1: other player won, -1: ai won, 2: tie}
        
        
        exp = {"state": "in progress"}
        reward = 0
        done = 0
        
        # illegal move
        if player == -1: #check for ai player only
            if prev_board[action] != 0 and player == -1:
                reward = -10  # illegal moves are really bad
                exp = {"state": "revisited", "reason": "Illegal move"}
                self.summary["illegal moves"] += 1
                
                return self.one_hot_board(), reward, exp
        
        self.board[action] = player

        # if the other player can win on the next turn and still ai is not being defensive
        for streak in self.winning_streaks:
            if ((self.board[streak] == 1).sum() == 2) and (self.board[streak] == 0).any():
                if player == -1:
                    reward = -2
                    exp = {
                        "state": "in progress",
                        "reason": "Player ai can lose on the next turn"
                    }

        for streak in self.winning_streaks:
            # check if ai player won
            if (self.board[streak] == -1).all():
                reward = 2
                exp = {
                    "state": "End",
                    "reason": "Player ai has won"
                }
                self.summary["total games"] += 1
                self.summary["player ai wins"] += 1
                done = -1
                break

                # check if other player won
            elif (self.board[streak] == 1).all(): 
                exp = {
                    "state": "End",
                    "reason": "Player other has won"
                }
                self.summary["total games"] += 1
                self.summary["player other wins"] += 1
                done = 1
                break
        
        # if game is drawn
        if (self.board != 0).all() and not done:
            reward = 0
            exp = {
                "state": "End",
                "reason": "Game Draw!"
            }
            done = 2
            self.summary["total games"] += 1
            self.summary["ties"] += 1

        # print("S5", self.one_hot_board(), reward, done, exp)
    
        return self.one_hot_board(), reward, done, exp

    def render(self, mode: str = "human"):
        print("{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}".format(*self.board.tolist()))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Policy(nn.Module):

    def __init__(self, n_inputs= 3*9, n_outputs=9):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def valid_actions(self, state):
        state = state.reshape(3, 3, 3)
        open_spots = state[:, :, 0].reshape(-1)
        return open_spots

    def act(self, state):
        with torch.no_grad():
            action1 = self.forward(state).max(1)[1].view(1, 1)

            actions = self.forward(state)
            actions, act_idx = actions[0].sort(descending = True)
            opens = self.valid_actions(state)
            
            for idx in act_idx:
                if opens[idx] == 1:
                    return action1, torch.tensor([[idx]], dtype=torch.long)



class Game:
    '''
    make board
    get player move
    update state
    feed state into model
    get action from model
    update state
    '''
    # def __init__(self, policy = Policy):
        # self.policy = Policy(n_inputs=3*9, n_outputs=9)
        
    # def player_move(self, board, )

    def load_model(self, path: str):
        model = Policy(n_inputs=3*9, n_outputs=9)
        model_state_dict = torch.load(path)
        model.load_state_dict(model_state_dict)
        model.eval()
        return model


    def select_dummy_action(self, state: np.array):
        state = state.reshape(3, 3, 3)
        open_spots = state[:, :, 0].reshape(-1)
        p = open_spots / open_spots.sum()
        r = np.random.choice(np.arange(9), p=p)
        return r
    
    def select_model_action(self, model: Policy, state: torch.tensor, eps: float):
        sample = random.random()
        if sample > eps:
            return model.act(state), False
        else:
            state = state.numpy()
            a = torch.tensor([[random.randrange(0, 9)]],dtype=torch.long,)
            b = torch.tensor([[self.select_dummy_action(state)]], dtype=torch.long)
            
            return (a, b), True

    def optimize_model(self, 
                        optimizer: optim.Optimizer,
                        policy: Policy,
                        target: Policy,
                        memory: ReplayMemory,
                        batch_size: int,
                        gamma: float):
        # from pytorch docs: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = memory.transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool,)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size)
        next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # print(loss)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    
        
        