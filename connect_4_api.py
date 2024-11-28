import dill
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from websockets.headers import parse_extension_item

import m_connect_4 as c4
import json


class Game:
    def __init__(self):
        # Initialize the Connect4 environment
        self.env = c4.Connect4Env()

        # Set state
        self.state = self.env.reset()

        # Create the Q-learning agent
        self.agent = c4.QLearningAgent(self.env, 0.9, 0.1, 0.8)

        # Load_q_table
        self.q_table = load_q_table()

    def reset(self):
        self.state = self.env.reset()

    def update_state(self, new_state):
        self.state = new_state

def load_q_table(filename="connect4_q_table.pkl"):
    """
    Load a Q-table using dill.

    Args:
        filename (str): The file name from which to load the Q-table.

    Returns:
        dict: The loaded Q-table.
    """
    with open(filename, "rb") as file:
        q_table = dill.load(file)
    print(f"Q-table successfully loaded from {filename}.")
    return q_table


game = Game()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/new_game")
async def new_game(player: int):
    game.reset()

    if player == 2:
        #exec agent action
        action = game.agent.choose_action(game.state)
        _, next_state, done = game.env.do_action(action)
        game.update_state(next_state)

    board = game.env.get_board()

    data = {'board': board}
    json_data = json.dumps(data)
    return json_data


@app.post("/play_move")
async def play_move(action: int):
    possible_actions = game.env.get_possible_actions(game.state)
    if action not in possible_actions:
        print('no actions')
        return [],


    #Do player action
    _, next_state, done = game.env.do_action(action)

    #Check if game is done and return winner
    if done:
        board = game.env.get_board()
        winner = int(game.env.check_winner(game.env.board))
        data = {'board': board, 'done': done, 'winner': winner}
        json_data = json.dumps(data)
        return json_data

    # Update current state
    game.update_state(next_state)

    #Do agent action
    action = game.agent.choose_action(game.state)
    reward, next_state, done = game.env.do_action(action)

    # Check if game is done and return winner
    if done:
        board = game.env.get_board()
        winner = game.env.check_winner(game.env.board)
        data = {'board': board, 'done': done, 'winner': winner}
        json_data = json.dumps(data)
        return json_data

    #Update current state and returns board
    game.update_state(next_state)
    board = game.env.get_board()

    data = {'board': board, 'done': False, 'winner': -1}
    json_data = json.dumps(data)
    return json_data