import numpy as np
import gym
import random
import time
import tensorflow as tf
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


class CycleBalancingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(9, ), dtype=np.uint8)
        self.np_random, _ = gym.utils.seeding.np_random()
        self.x = 0
        self.y = 0
        self.length = 100
        self.results = np.array([])
        self.epsilon = 0.2000 # Probability with which we select the random move (1-epsilon is the probability with which we select best move)

    def step(self, action):

        # Take another action if the output square is already filled
        if self.pos[action] !=0:
            return self.pos, -0.1, 0, {}


        x = (action//3)*100     # X-position of the action
        y = (action % 3)*100    # Y-position of the action
        #self.board[x+25: x+75, y+25:y+75] = self.zero   # Draw zero on the board_image
        self.pos[x//100*3+y//100] = 1   # Mark the postion on the board

        res = self.result() # Check if the game has ended
        if res == 3: # Draw
            print("Draw")
            self.results = np.append(self.results, res)
            if self.results.shape[0]>self.length:
                self.results = self.results[1:]
            self.print_result()
            return self.pos, 2, 1, {}
        elif res == 1: # Agent Won
            print("Win")
            self.results = np.append(self.results, res)
            if self.results.shape[0] > self.length:
                self.results = self.results[1:]
            self.print_result()
            return self.pos, 10, 1, {}

        if self.epsilon < random.random():
            best_move = self.get_best_move()
        else:
            best_move = self.get_random_move()

        self.x = (best_move % 3) * 100  # Get X-position of the Human Player's move
        self.y = (best_move//3)*100 # Get Y-position of the Human Player's move

        #self.board[self.y+25: self.y+75, self.x+25: self.x+75] = self.cross # Draw Human Player's move on the board_image
        self.pos[self.y // 100 * 3 + self.x // 100] = 2 # Make the Human Player's move on the board

        res = self.result() # Check if the game is over
        if res == 3: # Draw
            print("Draw")
            self.results = np.append(self.results, res)
            if self.results.shape[0] > self.length:
                self.results = self.results[1:]
            self.print_result()
            return self.pos, 2, 1, {}
        elif res == 2: # Human Wins
            print("Loss")
            self.results = np.append(self.results, res)
            if self.results.shape[0] > self.length:
                self.results = self.results[1:]
            self.print_result()
            return self.pos, -10, 1, {}
        else:
            #print("Continue")
            return self.pos, 0, 0, {}

    def get_best_move(self):
        board = self.pos
        best_move = -1
        best_move_val = -100000
        for i in range(9):
            if board[i] != 0: continue
            board[i] = 2
            tmp = self.minimax(0, board, 1)
            if tmp >= best_move_val:
                best_move = i
                best_move_val = tmp
            board[i] = 0
        return best_move

    def get_random_move(self):
        moves_left = []
        for i in range(9):
            if self.pos[i] == 0:
                moves_left.append(i)
        best_move = moves_left[random.randint(0, len(moves_left) - 1)]
        return best_move

    def print_result(self):
        #self.total += 1
        #print("Results = ", self.results)
        print("Win% = ", sum(self.results==1)/self.results.shape[0], '\tLoss% = ', sum(self.results==2)/self.results.shape[0], "\tDraw% = ", sum(self.results==3)/self.results.shape[0])
        #print("Epsilon = ", self.epsilon)
        #cv2.imshow('img', self.board)
        #cv2.waitKey(1)

    def result_2(self, board):  # Result function used by the minimax algorithm
        if board[0] == board[1] == board[2] == 1 or \
                board[3] == board[4] == board[5] == 1 or \
                board[6] == board[7] == board[8] == 1 or \
                board[0] == board[3] == board[6] == 1 or \
                board[1] == board[4] == board[7] == 1 or \
                board[2] == board[5] == board[8] == 1 or \
                board[0] == board[4] == board[8] == 1 or \
                board[2] == board[4] == board[6] == 1:
            return 1
        if board[0] == board[1] == board[2] == 2 or \
                board[3] == board[4] == board[5] == 2 or \
                board[6] == board[7] == board[8] == 2 or \
                board[0] == board[3] == board[6] == 2 or \
                board[1] == board[4] == board[7] == 2 or \
                board[2] == board[5] == board[8] == 2 or \
                board[0] == board[4] == board[8] == 2 or \
                board[2] == board[4] == board[6] == 2:
            return 2
        if sum(board == 0) == 0: return 3 # Draw
        return 0

    def minimax(self, maxi, board, depth): # Minimax Algorithm to find the best move
        if self.result_2(board) != 0:
            if self.result_2(board) == 2:
                return 100
            elif self.result_2(board) == 1:
                return -100
            else:
                return 0

        score = []
        for i in range(9):
            if board[i] != 0: continue
            board[i] = int(maxi)+1
            #print("Maxi = ", int(maxi)+1)
            score.append(self.minimax(not maxi, board, depth+1))
            board[i] = 0

        if maxi:
            return max(score)-depth
        else:
            return min(score)-depth

    def result(self):   # Result function used to check if the game is over and who is the winner (in the "step" function)
        if self.pos[0]==self.pos[1]==self.pos[2]==1 or \
            self.pos[3]==self.pos[4]==self.pos[5]==1 or \
            self.pos[6]==self.pos[7]==self.pos[8]==1 or \
            self.pos[0]==self.pos[3]==self.pos[6]==1 or \
            self.pos[1]==self.pos[4]==self.pos[7]==1 or \
            self.pos[2]==self.pos[5]==self.pos[8]==1 or \
            self.pos[0]==self.pos[4]==self.pos[8]==1 or \
            self.pos[2]==self.pos[4]==self.pos[6]==1:
                return 1
        if self.pos[0]==self.pos[1]==self.pos[2]==2 or \
            self.pos[3]==self.pos[4]==self.pos[5]==2 or \
            self.pos[6]==self.pos[7]==self.pos[8]==2 or \
            self.pos[0]==self.pos[3]==self.pos[6]==2 or \
            self.pos[1]==self.pos[4]==self.pos[7]==2 or \
            self.pos[2]==self.pos[5]==self.pos[8]==2 or \
            self.pos[0]==self.pos[4]==self.pos[8]==2 or \
            self.pos[2]==self.pos[4]==self.pos[6]==2:
                return 2
        if sum(self.pos == 0) == 0: return 3
        return 0

    def reset(self):
        #self.board = cv2.imread('board.jpg')    # Empty board
        #self.zero = cv2.imread('zero.jpg')      # Agent's mark (zero)
        #self.cross = cv2.imread('cross.jpg')    # Human player's mark (cross)

        self.pos = np.zeros((9)) # Vector or Matrice of the board postions
        self.epsilon = max(0.1, self.epsilon-.00005)
        return self.pos

    def render(self, mode='human'):
        return 0

    def close(self):
        p.disconnect(self.client)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


env = CycleBalancingEnv()
env.reset()
nb_actions = env.action_space.n

print(env.observation_space.shape)
print(nb_actions)

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1, 9)))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('Softmax'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=20,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-2), metrics=['mae'])

#dqn.load_weights('dqn_{}_weights.h5f'.format('Tik Tak To'))

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
dqn.fit(env, nb_steps=20000, visualize=False, verbose=2)

# After training is done, we save the final weights.
#print('Saving Weights')
#dqn.save_weights('dqn_{}_weights.h5f'.format('Tik Tak To'), overwrite=True)


# print(dqn.model)
#
# print("Testing")
# for i in range(10):
#     done = False
#     obs = env.reset()
#     while not done:
#         obs = obs.reshape((1, 1, -1))
#         action =- dqn.model.predict(obs)[0]
#         for i in range(9):
#             if env.pos[i]!=0:
#                 action[i] = -1000000
#         action = np.argmax(action)
#         obs, reward, done, _ = env.step(action)
