############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import collections


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)

        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.target_network = Network(input_dimension=2, output_dimension=4)
        self.update_target()

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transitions):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transitions)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transitions):
        states = []
        rewards = []
        actions = []
        next_states = []
        for transition in transitions:
            state, discrete_action, reward, next_state = transition
            states.append(state)
            rewards.append([reward])
            actions.append(discrete_action)
            next_states.append(next_state)
        input_tensor = torch.tensor(states)
        output_tensor = torch.tensor(rewards)
        network_predictions_for_all_directions = self.q_network.forward(input_tensor)
        predictions_tensor = torch.gather(network_predictions_for_all_directions, 1, torch.LongTensor(actions))
        next_state_tensor = torch.tensor(next_states)
        network_predictions_for_all_directions_for_next_state = self.target_network.forward(next_state_tensor)
        max_next_state = torch.unsqueeze(torch.max(network_predictions_for_all_directions_for_next_state, 1).values, 1)
        loss = torch.nn.MSELoss()(predictions_tensor, output_tensor + 0.95 * max_next_state)
        return loss


class ReplayBuffer:

    def __init__(self):
        self.container = collections.deque([], 100000)

    def append_transition(self, transition):
        if (len(self.container) == self.container.maxlen):
            print("Buffer full, dropping: ")
            print(self.container[0])
            self.container.remove(self.container[0])
        self.container.append(transition)

    def sample(self, num):
        if (len(self.container) < num):
            return []
        else:
            random_indice = np.random.choice(len(self.container), num, replace=False)
            return [self.container[i] for i in random_indice]


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length (you will need to increase this)
        self.episode_length = 800
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        self.dqn = DQN()
        self.buffer = ReplayBuffer()
        self.epsilon = 1
        self.epsilon_limit = 0.5
        self.should_decay_epsilon = False

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            return True
        else:
            return False

    def epsilon_greedy_action(self, state, epsilon):
        input_tensor = torch.tensor(state)
        output_tensor = self.dqn.q_network.forward(input_tensor)
        output = output_tensor.detach().numpy()
        print("Prediction:")
        print(output)
        sorted_output = np.sort(output)
        max_action = np.where(output == sorted_output[-1])[0][0]
        second_max_action = np.where(output == sorted_output[-2])[0][0]
        p = [epsilon / 4] * 4
        p[max_action] += 9 * (1 - epsilon) / 10
        p[second_max_action] += (1 - epsilon) / 10
        action = np.random.choice(range(4), 1, p=p)
        return action

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:  # Move right
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        if discrete_action == 1:  # Move down
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        if discrete_action == 2:  # Move left
            continuous_action = np.array([-0.02, 0], dtype=np.float32)
        if discrete_action == 3:  # Move up
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        return continuous_action

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        if (self.has_finished_episode()):
            if (self.should_decay_epsilon):
                if(self.episode_length >= 500):
                    self.episode_length -= 50
                if (self.epsilon_limit >= 0.1):
                    self.epsilon_limit -= 0.1
            self.should_decay_epsilon = False
        action = self.epsilon_greedy_action(state, self.epsilon)
        if (self.epsilon - 0.000015 <= self.epsilon_limit):
            self.epsilon = self.epsilon_limit
        else:
            self.epsilon -= 0.000015
        print(self.epsilon)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return self._discrete_action_to_continuous(action)

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        if (distance_to_goal <= 0.03):
            self.should_decay_epsilon = True
        # Convert the distance to a reward
        reward = 1 - distance_to_goal - 0.2
        if (distance_to_goal < 0.1):
            reward = 3 - distance_to_goal
        elif (distance_to_goal < 0.3):
            reward = 2.5 - distance_to_goal
        elif (distance_to_goal < 0.5):
            reward = 2 - distance_to_goal
        elif (np.array_equal(self.state, next_state)):
            reward = -1.5
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # Now you can do something with this transition ...
        self.buffer.append_transition(transition)
        mini_batch = self.buffer.sample(150)

        if (len(mini_batch)):
            loss = self.dqn.train_q_network(mini_batch)
        if (self.has_finished_episode()):
            self.dqn.update_target()

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        input_tensor = torch.tensor(state)
        output_tensor = self.dqn.q_network.forward(input_tensor)
        output = output_tensor.detach().numpy()
        print("Prediction:")
        print(output)
        sorted_output = np.sort(output)
        max_action = np.where(output == np.max(sorted_output))[0][0]
        return self._discrete_action_to_continuous(max_action)
