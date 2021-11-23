# ttt-on-spartificial

A Deep Q Network based reinforcement learning agent that learns from user interation.

## Demo

Initially agent plays random moves due to extensive exploration. 

Demo - First 10000 points


## How does it work

Reinforcement Learning

At first, the agent will play random moves, saving the states and the given reward in a limited queue (replay memory). At the end of each episode (game), the agent will train itself (using a neural network) with a random sample of the replay memory. As more and more games are played, the agent becomes smarter, achieving higher chance of winning.

Since in reinforcement learning once an agent discovers a good 'path' it will stick with it, it was also considered an exploration variable (that decreases over time), so that the agent picks sometimes a random action instead of the one it considers the best. This way, it can discover new 'paths' to achieve higher scores.
Training

The training is based on the Q Learning algorithm. Instead of using just the current state and reward obtained to train the network, it is used Q Learning (that considers the transition from the current state to the future one) to find out what is the best possible score of all the given states considering the future rewards, i.e., the algorithm is not greedy. This allows for the agent to take some moves that might not give an immediate reward, so it can get a bigger one later on (e.g. waiting to clear multiple lines instead of a single one).

The neural network will be updated with the given data (considering a play with reward reward that moves from state to next_state, the latter having an expected value of Q_next_state, found using the prediction from the neural network)

## results

Since the model is not trained at first and it only learn with human interaction, it can take a lot of time to learn and optimize it's actions. From our testing, we are hoping to get best results after 30_000 total game iterations. 

Link: https://spartificial.com/resources/game 


