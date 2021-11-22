from maingame import Game, TicTacToe, Policy, ReplayMemory 
import numpy as np
import torch
import torch.optim as optim
from flask import Flask, jsonify, request
import json
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

batch_size = 128
gamma = 0.99
eps_start = 1.0
eps_end = 0.1
eps_steps = 30_000

game = Game()
policy = Policy(n_inputs=3 * 9, n_outputs=9)

policy = game.load_model("test_weights.pt")

target = Policy(n_inputs=3 * 9, n_outputs=9)
target.load_state_dict(policy.state_dict())
target.eval()

optimizer = optim.Adam(policy.parameters(), lr=1e-3)
memory = ReplayMemory(50_000)

env = TicTacToe()
state = env.reset()
summaries = []



@app.route('/move', methods=['POST'])
def play():
    # 1 is user, -1 is ai

    game_num = env.summary['total games']
    
    t = np.clip(game_num / eps_steps, 0, 1)
    eps = (1 - t) * eps_start + t * eps_end
    # eps = 0.3
    if game_num > 30_000: 
        eps = 0
    
    # player (1) user goes ############################################################

    post = request.get_json()

    env.board = np.array(post.get("board"), dtype = "int")    # previous board from user (flatten)
    user_action = int(post.get("user_action"))             # action on previous board from user 

    next_state, _, done, _ = env.step(user_action, env.board, 1)

    if done == 1: #user won
        game.optimize_model(
            optimizer=optimizer,
            policy=policy,
            target=target,
            memory=memory,
            batch_size=batch_size,
            gamma=gamma,
        )
        if game_num % 1000 == 0:
            target.load_state_dict(policy.state_dict())
        torch.save(policy.state_dict(), "test_weights.pt")
        return jsonify(tied = False,
                        ai_wins = False,
                        user_wins = True,
                        board = json.dumps(list(map(int, env.board))),
                        summary = env.summary
                        )

    if done == 2: #tie
        game.optimize_model(
            optimizer=optimizer,
            policy=policy,
            target=target,
            memory=memory,
            batch_size=batch_size,
            gamma=gamma,
        )
        if game_num % 1000 == 0:
            target.load_state_dict(policy.state_dict())
        torch.save(policy.state_dict(), "test_weights.pt")
        return jsonify(tied = True,
                        ai_wins = False,
                        user_wins = False,
                        board = json.dumps(list(map(int, env.board))),
                        summary = env.summary)

    state = torch.tensor([next_state], dtype=torch.float)

     # player ai (-1) goes ---------------------------------------------------------------
    if done == 0: 
        action, was_random = game.select_model_action(policy, state, eps)        #this function requires state as tensor
        if was_random:
            env.summary["random actions"] += 1
            # next_state, reward, done, _ = env.step(action.item(),env.board,-1)
            # if done:
            #     next_state = None
            # else:
            #     next_state = torch.tensor([next_state], dtype=torch.float).to(device)

        # else:
        action1, action = action
        if env.board[action1.item()] != 0:
            next_state, reward, _ = env.step(action1.item(), env.board, -1)
            next_state = torch.tensor([next_state], dtype=torch.float)
            memory.push(state, action1, next_state, torch.tensor([reward]))
            
        next_state, reward, done, _ = env.step(action.item(), env.board, -1)
        if done == -1:
            next_state = None
        if done == 0:
            next_state = torch.tensor([next_state], dtype=torch.float)

        memory.push(state, action, next_state, torch.tensor([reward]))
    
    #ai won
    if done == -1: 
        game.optimize_model(
            optimizer=optimizer,
            policy=policy,
            target=target,
            memory=memory,
            batch_size=batch_size,
            gamma=gamma,
        )
        if game_num % 1000 == 0:
            target.load_state_dict(policy.state_dict())
        torch.save(policy.state_dict(), "test_weights.pt")
        return jsonify(tied = False,
                        ai_wins = True,
                        user_wins = False,
                        board = json.dumps(list(map(int, env.board))),
                        summary = env.summary)
    
    # game continues   
    if done == 0: 
        game.optimize_model(
            optimizer=optimizer,
            policy=policy,
            target=target,
            memory=memory,
            batch_size=batch_size,
            gamma=gamma,
        )
        if game_num % 1000 == 0:
            target.load_state_dict(policy.state_dict())
        torch.save(policy.state_dict(), "test_weights.pt")
        return jsonify(tied = False,
                        ai_wins = False,
                        user_wins = False,
                        board = json.dumps(list(map(int, env.board))),
                        summary = env.summary)
