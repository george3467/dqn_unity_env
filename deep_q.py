import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import mlagents
import mlagents_envs
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


def get_environment(file_path):
    """
    This function imports the unity environment, converts it to a 
    gym environment, and calculates the number of actions and observations.
    """

    # imports the Unity environment
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_path, side_channels=[channel])

    # sets the window size for the environment
    channel.set_configuration_parameters(width=2000, height=1000)

    # converts the Unity environment into a Gym environment
    env = UnityToGymWrapper(env, allow_multiple_obs=True)

    # obtains the number of actions and number of the observations
    num_actions = env.action_space.n
    state_size = len(env.reset()[0])

    return env, state_size, num_actions


class Q_Model(nn.Module):
    """
    This model is the Deep Q-Network that will be trained to predict the
    actions of the agent. It is a simple sequence of Linear layers with
    Tanh activations between the layers.
    """
    def __init__(self, state_size, num_actions):
        super(Q_Model, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, num_actions),
        )
    def forward(self, x):
        return self.linear(x)



Data = namedtuple('Data', ('state', 'action', 'reward', 'next_state'))
class ReplayBuffer():
    """
    This buffer is used to store the data obtained as the agent moves around
    in the Gym environment. The data is stored in the format desribed by the
    namedtuple "Data." In the training steps, data is sampled from this buffer
    to train the policy_model.
    """
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    # adds new data to the buffer
    def push(self, *args):
        self.buffer.append(Data(*args))

    # samples data from the buffer for training
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)



def get_action(policy_model, state, num_actions, epsilon):
    """
    This function returns the next action for the agent. Epsilon defines the probability
    of how often this function should return a random action. 
    """
    sample = random.random()

    # choose a random action
    if sample < epsilon:
        action = torch.distributions.Categorical(torch.ones(num_actions)).sample().reshape(1, 1)

    # predict an action using the policy_model
    else:
        with torch.no_grad():
            action_logits = policy_model(state)

            # choose the action with the highest logit
            action = torch.argmax(action_logits).reshape(1, 1)
    return action



def train_step(policy_model, target_model, memory_buffer, optimizer, batch_size):
    """
    This function performs one optimization step according to the Deep Q-Learning method. 
    """

    if len(memory_buffer.buffer) < batch_size:
        return
    
    # sample training data from the buffer
    dataset = memory_buffer.sample(batch_size)

    batch = Data(*zip(*dataset))
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_model(state_batch).gather(1, action_batch)

    # skip instances where next_state = None
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # calculate expected values
    next_state_values = torch.zeros(batch_size)
    with torch.no_grad():
        target_values = target_model(non_final_next_states)
        next_state_values[non_final_mask] = torch.max(target_values, dim=-1).values
    expected_state_action_values = reward_batch + 0.99 * next_state_values

    # loss calculation and optimzation
    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_model.parameters(), 100)
    optimizer.step()



def run_training():
    """
    This script trains the policy_model. 
    """

    # obtain the Gym environment
    file_path = "Unity_Environment.app"
    env, state_size, num_actions = get_environment(file_path)

    # initialize the Policy ang Target models
    policy_model = Q_Model(state_size, num_actions)
    target_model = Q_Model(state_size, num_actions)
    target_model.load_state_dict(policy_model.state_dict())

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-4, amsgrad=True)
    memory_buffer = ReplayBuffer(10_000)

    batch_size = 128
    tau = 0.005
    steps_per_episode = 500

    # total number of episodes to run
    num_episodes = 900

    # number of episodes over which the epsilon will be decayed
    decay_episodes = 600

    for i_episode in range(num_episodes):
        state = env.reset()
        state = torch.asarray(state, dtype=torch.float32)

        # epsilon is decayed linearly
        if i_episode < decay_episodes:
            epsilon = 0.9 * (decay_episodes - i_episode)/decay_episodes + 0.05

        # after decay episodes are completed
        else:
            epsilon = 0.05

        for t in range(steps_per_episode):
            action = get_action(policy_model, state, num_actions, epsilon)
            next_state, reward, done, _ = env.step(action.item())
            reward = torch.asarray([reward], dtype=torch.float32)
            if done:
                next_state = None
            else:
                next_state = torch.asarray(next_state, dtype=torch.float32)
            
            # save the data from the current step
            memory_buffer.push(state, action, reward, next_state)
            
            # perform one optimization step
            train_step(policy_model, target_model, memory_buffer, optimizer, batch_size)

            state = next_state

            # soft update of the target_model
            target_model_state_dict = target_model.state_dict()
            policy_model_state_dict = policy_model.state_dict()
            for key in policy_model_state_dict:
                target_model_state_dict[key] = policy_model_state_dict[key]*tau + target_model_state_dict[key]*(1-tau)
            target_model.load_state_dict(target_model_state_dict)

            if done:
                print(i_episode, " done in: ", t, epsilon)
                break

    env.close()    
    torch.save(policy_model.state_dict(), 'unity_weights_1.pth')



def run_inference():
    """
    This script runs inference for the trained policy_model.
    """

    # obtain the Gym environment
    file_path = "Unity_Environment.app"
    env, state_size, num_actions = get_environment(file_path)

    # initialize the model and load the trained weights
    policy_model = Q_Model(state_size, num_actions)
    policy_model.load_state_dict(torch.load('unity_weights.pth'))

    num_iterations = 30
    steps_per_episode = 500

    for i in range(num_iterations):
        state = env.reset()
        state = torch.asarray(state, dtype=torch.float32)
        
        for t in range(steps_per_episode):
            action_logits = policy_model(state)
            action = torch.argmax(action_logits)
            state, _, done, _ = env.step(action)
            state = torch.asarray(state, dtype=torch.float32)
            if done:
                print(i+1, "done in ", t+1) 
                break

    env.close()





