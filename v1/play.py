import numpy as np
import gym
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

    def draw_circle(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print('Mouse postion = ', x, y)
            self.x = x
            self.y = y
            return 0
        return 1

    def step(self, action):
        # print(action)
        # action = np.argmax(action)
        # print(action)

        if self.pos[action]!=0: return self.pos, -1, 0, {}

        x = (action//3)*100
        y = (action % 3)*100
        self.board[x+25: x+75, y+25:y+75] = self.zero
        self.pos[x//100*3+y//100] = 1
        cv2.imshow('img', self.board)
        cv2.waitKey(1)
        #print(self.pos)
        res = self.result()
        if res == 3:
            print("Draw")
            cv2.waitKey(0)
            return self.pos, 2, 1, {}
        elif res == 1:
            print("Win")
            cv2.waitKey(0)
            return self.pos, 10, 1, {}
        elif res == 2:
            print("Loss")
            cv2.waitKey(0)
            return self.pos, -10, 1, {}

        cv2.namedWindow('img')
        cv2.setMouseCallback('img', self.draw_circle)
        cv2.waitKey(0)
        self.x = (self.x//100)*100
        self.y = (self.y//100)*100

        self.board[self.y+25: self.y+75, self.x+25: self.x+75] = self.cross
        self.pos[self.y // 100 * 3 + self.x // 100] = 2
        cv2.imshow('img', self.board)
        cv2.waitKey(1)
        #print(self.pos)
        res = self.result()
        if res == 3:
            print("Draw")
            cv2.waitKey(0)
            return self.pos, 2, 1, {}
        elif res == 1:
            print("Win")
            cv2.waitKey(0)
            return self.pos, 10, 1, {}
        elif res == 2:
            print("Loss")
            cv2.waitKey(0)
            return self.pos, -10, 1, {}
        else:
            print("Continue")
            return self.pos, 0, 0, {}

    def result(self):
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
        self.board = cv2.imread('board.jpg')
        self.cross = cv2.imread('cross.jpg')
        self.zero = cv2.imread('zero.jpg')

        cv2.imshow('img', self.board)
        cv2.waitKey(1)
        self.pos = np.zeros((9))
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
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.load_weights('dqn_{}_weights.h5f'.format('Tik Tak To'))

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
dqn.fit(env, nb_steps=100, visualize=False, verbose=2)

# After training is done, we save the final weights.
#dqn.save_weights('dqn_{}_weights.h5f'.format('Tik Tak To'), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
#dqn.test(env, nb_episodes=5, visualize=False, verbose=2)
