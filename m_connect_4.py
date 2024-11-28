import numpy as np
import random
from collections import defaultdict
import dill
from datetime import datetime,timezone



# Environment for Connect4
class Connect4Env:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)  # 0: empty, 1: red, 2: yellow
        self.current_player = 1  # Red starts
        self.state = self.get_state()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.state = self.get_state()
        return self.state

    def get_state(self):
        # State can be represented as a tuple of tuples for easier indexing in Q-table
        return tuple(map(tuple, self.board))

    def is_valid_move(self, col):
        return self.board[0, col] == 0  # Check if the column is not full

    def get_possible_actions(self, state):
        # Actions are columns where a move can be made
        return [col for col in range(self.cols) if self.is_valid_move(col)]

    def get_next_available_row(self, col):
        for row in reversed(range(self.rows)):
            if self.board[row,col] == 0:
                return row

    def play_move(self, col):
        if not self.is_valid_move(col):
            return False  # Invalid move

        row = self.get_next_available_row(col)
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
        self.state = self.get_state()
        return True

    def switch_player(self):
        self.current_player = 3 - self.current_player  # Switches between 1 (red) and 2 (yellow)

    def check_winner(self, board):
        # Check all possible winning conditions, returns 1 if red wins, 2 if yellow wins, 0 if no winner
        for row in range(self.rows):
            for col in range(self.cols):
                if board[row][col] == 0:
                    continue
                if col <= self.cols - 4 and all(board[row][col+i] == board[row][col] for i in range(4)):
                    return board[row][col]
                if row <= self.rows - 4 and all(board[row+i][col] == board[row][col] for i in range(4)):
                    return board[row][col]
                if col <= self.cols - 4 and row <= self.rows - 4 and all(board[row+i][col+i] == board[row][col] for i in range(4)):
                    return board[row][col]
                if col >= 3 and row <= self.rows - 4 and all(board[row+i][col-i] == board[row][col] for i in range(4)):
                    return board[row][col]
        return 0

    def is_terminal(self, state):
        return self.check_winner(self.board) != 0 or all(self.board[0, col] != 0 for col in range(self.cols))

    def simulate_action(self, action):
        col = action

        if not self.is_valid_move(col):
            return False  # Invalid move

        row = self.get_next_available_row(col)
        new_board = self.board.copy()

        if self.board[row, col] == 0:
            new_board[row,col] = self.current_player

        next_state = tuple(map(tuple, new_board))

        winner = self.check_winner(new_board)

        if winner == 1:  # Reward for red win
            return 100, next_state, True
        elif winner == 2:  # Reward for yellow win
            return -100, next_state, True
        elif self.is_terminal(next_state):  # Tie
            return 0, next_state, True
        else:
            return -1, next_state, False


    def do_action(self, action):
        # Executes an action, returns reward, next state, and whether it's done
        self.play_move(action)

        winner = self.check_winner(self.board)
        if winner == 1:  # Reward for red win
            return 100, self.get_state(), True
        elif winner == 2:  # Reward for yellow win
            return -100, self.get_state(), True
        elif self.is_terminal(self.state):  # Tie
            return 0, self.get_state(), True
        else:
            self.switch_player()
            return -1, self.get_state(), False  # Small penalty to encourage faster wins

    def print_board(self):
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                if self.board[i,j] == 1:
                    row.append('X')
                elif self.board[i,j] == 2:
                    row.append('O')
                else:
                    row.append(' ')

            print('|', row, '|')

        print('|   0    1    2    3    4    5    6   |')
        print('       ')
        print('       ')

    def get_board(self):
        board = []

        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                if self.board[i, j] == 1:
                    row.append(1)
                elif self.board[i, j] == 2:
                    row.append(2)
                else:
                    row.append(0)

            board.append(row)

        return board

class QLearningAgent:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma  # Discount factor
        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Exploration rate
        self.q_table = defaultdict(lambda: np.zeros(self.env.cols))  # Q-values initialized to 0
        self.state = self.env.reset()

    def choose_action(self, state):
        """
        Chooses an action using an epsilon-greedy policy.
        """
        if random.random() < self.epsilon:  # Explore
            possible_actions = self.env.get_possible_actions(state)
            return random.choice(possible_actions) if possible_actions else None
        else:  # Exploit
            q_values = self.q_table[state]
            possible_actions = self.env.get_possible_actions(state)
            best_action = max(possible_actions, key=lambda a: q_values[a]) if possible_actions else None
            return best_action

    def update_q_value(self, state, action, reward, next_state):
        """
        Updates the Q-value for a given state-action pair.
        """
        max_future_q = 0
        possible_actions = self.env.get_possible_actions(next_state)
        if possible_actions:
            max_future_q = max(self.q_table[next_state][a] for a in possible_actions)

        # Q-learning update rule
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_future_q - self.q_table[state][action])

    def run_q_learning(self, num_episodes=10000):
        """
        Runs the Q-learning algorithm for a specified number of episodes.
        """

        exec_times = []
        init_time = datetime.now(timezone.utc)
        for episode in range(num_episodes):
            state = self.env.reset()  # Reset environment for a new episode
            done = False

            while not done:

                action = self.choose_action(state)  # Select action using epsilon-greedy policy
                if action is None:  # No valid actions (e.g., terminal state)
                    break

                reward, next_state, done = self.env.do_action(action)  # Perform action
                self.update_q_value(state, action, reward, next_state)  # Update Q-value
                state = next_state  # Move to next state

        done_time = datetime.now(timezone.utc)
        diff = done_time - init_time
        exec_times.append(diff.seconds)
        print('Exec time (secs)=', diff.seconds)
        print(exec_times)

        return self.q_table

    def get_policy(self):
        """
        Derives the policy from the Q-table (greedy policy).
        """
        policy = {}
        for state, q_values in self.q_table.items():
            possible_actions = self.env.get_possible_actions(state)
            if possible_actions:
                best_action = max(possible_actions, key=lambda a: q_values[a])
                policy[state] = best_action
        return policy

def save_q_table(q_table, filename="q_table.pkl"):
    """
    Save the Q-table using dill.

    Args:
        q_table (dict): The Q-table to save.
        filename (str): The file name to save the Q-table.
    """
    with open(filename, "wb") as file:
        dill.dump(q_table, file)
    print(f"Q-table successfully saved to {filename}.")

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

def test_agent_vs_human(agent, env, num_games=1):
    wins = 0
    losses = 0
    ties = 0

    for _ in range(num_games):
        state = env.reset()  # Reset the environment for a new game
        done = False
        while not done:
            # Q-learning agent's turn
            action = agent.choose_action(state)
            if action is None:  # No valid actions (e.g., terminal state)
                break
            reward, next_state, done = env.do_action(action)

            env.print_board()

            if done:  # Check the game result after the agent's turn
                env.print_board()

                if reward == 100:  # Q-learning agent wins
                    wins += 1
                elif reward == -100:  # Random agent wins
                    losses += 1
                else:  # Tie
                    ties += 1
                break

            # Random opponent's turn
            possible_actions = env.get_possible_actions(state)
            input_text = "Select an action from: "
            for a in possible_actions:
                input_text+= str(a) + " "

            input_text += " -> "
            human_action = input(input_text)

            print(human_action)
            while int(human_action) not in possible_actions:
                human_action = input(input_text)

            _, next_state, done = env.do_action(int(human_action))

            env.print_board()
            if done:  # Check the game result after the opponent's turn
                env.print_board()
                if env.check_winner(env.board) == 2:  # Opponent wins
                    losses += 1
                elif env.is_terminal(env.state):  # Tie
                    ties += 1
                break

            state = next_state  # Update the state

    total_games = wins + losses + ties
    print(f"Out of {total_games} games:")
    print(f"Q-Learning Agent Wins: {wins} ({(wins / total_games) * 100:.2f}%)")
    print(f"Human Opponent Wins: {losses} ({(losses / total_games) * 100:.2f}%)")
    print(f"Ties: {ties} ({(ties / total_games) * 100:.2f}%)")

import json
if __name__ == '__main__':

    # Initialize the Connect4 environment
    env = Connect4Env()

    # Create the Q-learning agent
    agent = QLearningAgent(env, 0.9, 0.1, 0.8)

    # Train the agent
    q_table = agent.run_q_learning(num_episodes=400000)
    #q_table = load_q_table("connect4_q_table.pkl")

    save_q_table(q_table, filename="connect4_q_table.pkl")

    # Test the agent after training
    #test_agent_vs_human(agent, env, num_games=2)
