import tqdm
import gym
from gym import spaces
import d4rl
import torch

import numpy as np

class OptionWrapper(gym.Wrapper):
    """Augments the actions of the environment with options from the HSSM.

    The wrapped environment must have a discrete action space.
    """

    def __init__(self, env, hssm, train_loader, seq_size, init_size,
                 threshold=0.05, recurrent=False):
        super().__init__(TensorWrapper(env))

        self._hssm = hssm

        # Compute z's on which HSSM has support over threshold:
        # i.e., {z | p(z) > threshold}
        num_options = hssm.post_abs_state.latent_n
        # p(z): of shape (num_options,)
        marginal = np.zeros(num_options)

        for train_obs_list, train_action_list in tqdm.tqdm(train_loader):
            # Mean of the marginals in the batch, where each batch is
            # weighted by batch_size / total dataset size
            weight = train_obs_list.shape[0] / len(train_loader.dataset)
            marginal += hssm.abs_marginal(
                    train_obs_list, train_action_list, seq_size,
                    init_size)[0].cpu().data.numpy() * weight

            del train_obs_list
            del train_action_list

        self._permitted_zs = [z for z in range(num_options)
                              if marginal[z] >= threshold]

        # Action space is default low-level actions + options
        self.action_space = spaces.Discrete(
                env.action_space.n + len(self._permitted_zs))

        self._current_state = None
        self._boundary_state = None
        self._recurrent = recurrent

    def step(self, action):
        # Default low-level actions
        if action < self.env.action_space.n:
            state, reward, done, info = self.env.step(action)
            self._current_state = state
            return state, reward, done, info

        # Taking an option as an action
        # Follows the option until the option terminates
        z = self._permitted_zs[action - self.env.action_space.n]
        state = self._current_state
        total_reward = 0  # accumulate all rewards during option
        low_level_actions = []
        self._boundary_state = self._hssm.initial_boundary_state(state[0].float())
        hidden_state = None
        while True:
            action, next_hidden_state = self._hssm.play_z(
                    z, state[0].float(), hidden_state,
                    recurrent=self._recurrent)
            next_state, reward, done, info = self.env.step(action)
            low_level_actions.append(action)
            total_reward += reward
            terminate, self._boundary_state = self._hssm.z_terminates(
                    next_state[0].float(), action, self._boundary_state)
            state = next_state
            hidden_state = next_hidden_state
            if done or terminate:
                break

        self._current_state = state
        info["low_level_actions"] = low_level_actions
        info["steps"] = len(low_level_actions)
        return state, total_reward, done, info

    def reset(self):
        self._current_state = self.env.reset()
        return self._current_state


class TensorWrapper(gym.ObservationWrapper):
    def observation(self, state):
        return torch.tensor(state[0]), torch.tensor(state[1])
    
class TensorWrapperD4RL(gym.ObservationWrapper):
    def observation(self, state):
        return torch.tensor(state, dtype=torch.float32)
    



class OracleOptionWrapper(gym.Wrapper):
    """Augments the actions of the environment with oracle options for grid."""

    def __init__(self, env):
        super().__init__(TensorWrapper(env))
        self.action_space = spaces.Discrete(env.action_space.n + 10)

    def step(self, action):
        # Default low-level actions
        if action < self.env.action_space.n:
            state, reward, done, info = self.env.step(action)
            self._current_state = state
            return state, reward, done, info

        # Take the option that picks up the nearest option_index-th object
        # If no such object exists, just moves up
        object_index = action - self.env.action_space.n

        def find_match(obj_type):
            for x in range(self.env.width):
                for y in range(self.env.height):
                    obj = self.env.get((x, y))
                    if obj is not None and obj.type == object_index:
                        return (x, y)
            return None

        match_pos = find_match(object_index)
        if match_pos is None:
            return self.env.step(0)

        num_steps = np.sum(np.abs(self.unwrapped.agent_pos - np.array(match_pos)))
        self.unwrapped._agent_pos = np.array(match_pos)

        next_state, reward, done, info = self.env.step(4)
        info["steps"] = num_steps + 1
        return next_state, reward, done, info

    def reset(self):
        return self.env.reset()


class OptionWrapperContinous(gym.Wrapper):
    """Augments the actions of the environment with options from the HSSM.

    The wrapped environment must have a discrete action space.
    """

    def __init__(self, env, hssm, train_loader, seq_size, init_size,
                 threshold=0.05, recurrent=False):
        super().__init__(TensorWrapperD4RL(env))

        self._hssm = hssm

        # Compute z's on which HSSM has support over threshold:
        # i.e., {z | p(z) > threshold}
        num_options = hssm.post_abs_state.latent_n
        # p(z): of shape (num_options,)
        marginal = np.zeros(num_options)

        for train_obs_list, train_action_list in tqdm.tqdm(train_loader):
            # Mean of the marginals in the batch, where each batch is
            # weighted by batch_size / total dataset size
            weight = train_obs_list.shape[0] / len(train_loader.dataset)
            marginal += hssm.abs_marginal(
                    train_obs_list, train_action_list, seq_size,
                    init_size)[0].cpu().data.numpy() * weight

            del train_obs_list
            del train_action_list

        self._permitted_zs = [z for z in range(num_options)
                              if marginal[z] >= threshold]

        # Action space is default low-level actions + options
        self.action_space = spaces.Discrete(
                 len(self._permitted_zs)+1, env.action_space.shape[0])

        self._current_state = None
        self._boundary_state = None
        self._recurrent = recurrent

    def step(self, action, frames):
        # if "tensor" in str(type(action)):
        #     action = action.cpu().detach().numpy()
        # Default low-level actions
        # compute the porbability of the complement
        # print(action)
        options_probs = self.softmax(action[:len(self._permitted_zs)+1])
        option_selection_prob = options_probs[:-1].sum()
        low_level_control_prob = 1 - option_selection_prob
        if low_level_control_prob > option_selection_prob:
            state, reward, done, info = self.env.step(action[len(self._permitted_zs)+1:])
            self._current_state = state
            return state, reward, done, info, frames

        # Taking an option as an action
        # Follows the option until the option terminates
        option = options_probs[:-1].argmax()
        z = self._permitted_zs[option]
        state = self._current_state
        total_reward = 0  # accumulate all rewards during option
        low_level_actions = []
        self._boundary_state = self._hssm.initial_boundary_state(state, action_dim=self.env.unwrapped.action_space.shape[0])
        hidden_state = None
        while True:
            action, next_hidden_state = self._hssm.play_z(
                    z, state, hidden_state,
                    recurrent=self._recurrent)
            next_state, reward, done, info = self.env.step(action.cpu().detach())
            frames.append(self.env.render("rgb_array"))
            low_level_actions.append(action)
            total_reward += reward
            terminate, self._boundary_state = self._hssm.z_terminates(
                    next_state, action, self._boundary_state)
            state = next_state
            hidden_state = next_hidden_state
            if done or terminate:
                break

        self._current_state = state
        info["low_level_actions"] = low_level_actions
        info["steps"] = len(low_level_actions)
        return state, total_reward, done, info, frames

    def reset(self):
        self._current_state = self.env.reset()
        return self._current_state
    
    def softmax(self, x):
        numerator = np.exp(x - np.max(x))
        denominator = np.sum(np.exp(x- np.max(x)))
        return numerator/denominator
        