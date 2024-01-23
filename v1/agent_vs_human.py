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
        self.win = 0 # Count number of Wins
        self.loss = 0 # Count number of losses
        self.draw = 0 # Count number of Draw games
        self.total = 0 # Count of the toatl number of games
        self.epsilon = 0.2000 # Probability with which we select the random move (1-epsilon is the probability with which we select best move)

    def step(self, action):

        # Take another action if the output square is already filled
        if self.pos[action] !=0:
            return self.pos, -1, 0, {}


        x = (action//3)*100     # X-position of the action
        y = (action % 3)*100    # Y-position of the action
        self.board[x+25: x+75, y+25:y+75] = self.zero   # Draw zero on the board_image
        self.pos[x//100*3+y//100] = 1   # Mark the postion on the board

        cv2.imshow('img', self.board)
        cv2.waitKey(1)

        res = self.result() # Check if the game has ended
        if res == 3: # Draw
            print("Draw")
            self.draw += 1
            self.print_result()
            return self.modify_pos(self.pos), 2, 1, {}
        elif res == 1: # Agent Won
            print("Win")
            self.win += 1
            self.print_result()
            return self.modify_pos(self.pos), 10, 1, {}

        print("Double click on the square you want to move")
        cv2.namedWindow('img')
        cv2.setMouseCallback('img', self.draw_circle)
        cv2.waitKey(0)
        self.x = (self.x//100)*100    # Get X-position of the Human Player's move
        self.y = (self.y//100)*100    # Get Y-position of the Human Player's move

        self.board[self.y+25: self.y+75, self.x+25: self.x+75] = self.cross # Draw Human Player's move on the board_image
        self.pos[self.y // 100 * 3 + self.x // 100] = 2 # Make the Human Player's move on the board

        cv2.imshow('img', self.board)
        cv2.waitKey(1)

        res = self.result() # Check if the game is over
        if res == 3: # Draw
            print("Draw")
            self.draw += 1
            self.print_result()
            return self.modify_pos(self.pos), 2, 1, {}
        elif res == 2: # Human Wins
            print("Loss")
            self.loss += 1
            self.print_result()
            return self.modify_pos(self.pos), -10, 1, {}
        else:
            #print("Continue")
            return self.modify_pos(self.pos), 0, 0, {}

    def modify_pos(self, pos):
        pos_tmp = pos.copy()
        pos_tmp[pos_tmp==2] = -1
        print(pos_tmp)
        print(self.pos)
        return pos_tmp

    def draw_circle(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print('Mouse postion = ', y//100, x//100)
            print("Press Enter Key...")
            self.x = x
            self.y = y
            return 0
        return 1

    def print_result(self):
        self.total += 1
        print("Win% = ", self.win / self.total, '\tLoss% = ', self.loss / self.total, "\tDraw% = ", self.draw / self.total)
        #print("Epsilon = ", self.epsilon)
        cv2.imshow('img', self.board)
        print("Press Enter Key to Continue to next game...")
        cv2.waitKey(0)

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
        self.board = cv2.imread('board.jpg')    # Empty board
        self.zero = cv2.imread('zero.jpg')      # Agent's mark (zero)
        self.cross = cv2.imread('cross.jpg')    # Human player's mark (cross)

        cv2.imshow('img', self.board)
        cv2.waitKey(1)

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
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

model.load_weights('dqn_{}_weights.h5f'.format('Tik Tak To'))

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1, enable_double_dqn=True, enable_dueling_network=True,
               dueling_type='avg', target_model_update=1e-2, policy=policy, gamma=0.9)
dqn.compile(Adam(lr=1e-5), metrics=['mae'])

dqn.load_weights('dqn_{}_weights.h5f'.format('Tik Tak To'))

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
dqn.fit(env, nb_steps=20000, visualize=False, verbose=2)

# After training is done, we save the final weights.
#dqn.save_weights('dqn_{}_weights.h5f'.format('Tik Tak To'), overwrite=True)


# for i in range(1000):
#     pos = get_current_board_postion()
#     action = dqn(pos)
#     env.perform_action(action)
#     env.make_human_move()
#
#     if env.done():
#         env.reset()
