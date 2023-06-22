import argparse
import collections
import os
import shutil
import time
import pickle

import git
import gym
import d4rl
# import gym_miniworld
import numpy as np
import torch
import tqdm

import config as cfg
import dqn
import dqn_utils
from grid_world import grid
import option_wrapper
import rl
import utils
# from world3d import world3d
import wandb
import mujoco_py


def eval(env, option, hssm, num_eps=10):
    returns = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for _ in range(num_eps):
        total_reward = 0
        state = torch.tensor(env.reset(), device=device, dtype=torch.float32)
        hidden_state = None
        for _ in range(1000):
            action, next_hidden_state = hssm.play_z(
                    option, state, hidden_state,
                    recurrent=False)
            next_state, reward, done, info = env.step(action.cpu().detach())
            total_reward += reward
            state = torch.tensor(next_state, device=device, dtype=torch.float32)
            hidden_state = next_hidden_state
        returns.append(total_reward)
    return np.mean(returns), np.std(returns)



def best_option(env_name, num_options, hssm, num_eps=10):
    env = gym.make(env_name)
    max_return = 0
    max_option = 0
    for option in range(num_options):
        average_return, return_std = eval(env, option, hssm, num_eps)
        if average_return > max_return:
            print(f"New max return:{average_return}, std:{return_std}, option:{option}")
            max_return = average_return
            max_option = option
    print(f"Best option:{max_option}, average return:{average_return}, std:{return_std}")
    return max_option

        
        



def run_episode(env, policy, experience_observers=None, test=False,
                return_render=False):
    """Runs a single episode on the environment following the policy.

    Args:
        env (gym.Environment): environment to run on.
        policy (Policy): policy to follow.
        experience_observers (list[Callable] | None): each observer is called with
            with each experience at each timestep.

    Returns:
        episode (list[Experience]): experiences from the episode.
    """
    def maybe_render(env, instruction, action, reward, info, timestep):
        if return_render:
            render = env.render()
            render.write_text("Action: {}".format(str(action)))
            render.write_text("Instruction: {}".format(instruction))
            render.write_text("Reward: {}".format(reward))
            render.write_text("Timestep: {}".format(timestep))
            render.write_text("Info: {}".format(info))
            return render
        return None

    if experience_observers is None:
        experience_observers = []

    episode = []
    state = env.reset()
    init_state = (np.array([2.90749422 , 4.92641686 ]), np.array([ 0.00, 0.00]))
        # init_state = (np.array([0.96808476, 6.07712179]), np.array([ 0.00, 0.00]))
    init_state = mujoco_py.cymj.MjSimState(time=0.0,
                                    qpos=init_state[0], qvel=init_state[1], act=None, udd_state={})
    env.sim.set_state(init_state)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    state = torch.tensor(np.array([2.90749422 , 4.92641686, 0.00, 0.00]).astype(np.float32), device=device)
    timestep = 0
    frames = [env.render("rgb_array")]
    # renders = [maybe_render(env, state[1], None, 0, {}, timestep)]
    hidden_state = None
    while True:
        # print(f"state in main:{state}")
        len_before = len(frames)
        action, next_hidden_state = policy.act(state, hidden_state, test=test)
        next_state, reward, done, info, frames = env.step(action, frames)
        timestep += 1
        # renders.append(maybe_render(env, next_state[1], action, reward, info, timestep))
        experience = rl.Experience(
                state, action, reward, next_state, done, info, hidden_state,
                next_hidden_state)
        episode.append(experience)
        for observer in experience_observers:
            observer(experience)

        if "experiences" in info:
            del info["experiences"]

        state = next_state
        hidden_state = next_hidden_state
        if len(frames) == len_before:
            frames.append(env.render("rgb_array"))
        if done:
            return episode, frames


def main(params=None, config_bindings=None):
    print("main has been called")
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument(
    #         '-c', '--configs', action='append', default=["configs/default.json"])
    # arg_parser.add_argument(
    #         '-b', '--config_bindings', action='append', default=[],
    #         help="bindings to overwrite in the configs.")
    # arg_parser.add_argument(
    #         "-x", "--base_dir", default="experiments",
    #         help="directory to log experiments")
    # arg_parser.add_argument(
    #         "-p", "--checkpoint", default=None,
    #         help="path to checkpoint directory to load from or None")
    # arg_parser.add_argument(
    #         "-f", "--force_overwrite", action="store_true",
    #         help="Overwrites experiment under this experiment name, if it exists.")
    # arg_parser.add_argument(
    #         "-s", "--seed", default=0, help="random seed to use.", type=int)
    # arg_parser.add_argument("exp_name", help="name of the experiment to run")
    # args = arg_parser.parse_args()
    args = {
        "configs": ["configs/default.json"],
        "config_bindings": [],
        "base_dir": "experiments",
        "checkpoint": None,
        "force_overwrite": False,
        "seed": 1,
        "exp_name": None, # This is a required argument
    }
    if params:
        for key in params:
            args[key] = params[key]
    name = args["exp_name"]
    # path = args["checkpoint"]
    # args["config_bindings"].append(f"checkpoint=\"{path}\"")
    print(f"name: {name}")
    # try:
    if config_bindings:
        args["config_bindings"]  = config_bindings
    assert args["exp_name"] is not None
    config = cfg.Config.from_files_and_bindings(
            args["configs"], args["config_bindings"])
    # except Exception as error:
    #     print(f"error:{error}")
    #     quit()

    
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    

    exp_dir = os.path.join(os.path.expanduser(args["base_dir"]), args["exp_name"])
    if os.path.exists(exp_dir) and not args["force_overwrite"]:
        raise ValueError("Experiment already exists at: {}".format(exp_dir))
    shutil.rmtree(exp_dir, ignore_errors=True)  # remove directory if exists
    time.sleep(5)
    os.makedirs(exp_dir)

    with open(os.path.join(exp_dir, "config.json"), "w+") as f:
        config.to_file(f)
    print(config)
    print(f"params:{params}")

    with open(os.path.join(exp_dir, "metadata.txt"), "w+") as f:
        repo = git.Repo()
        f.write("Commit: {}\n\n".format(repo.head.commit))
        commit = repo.head.commit
        diff = commit.diff(None, create_patch=True)
        for patch in diff:
            f.write(str(patch))
            f.write("\n\n")

    uid = wandb.util.generate_id()
    name = config.get("env")
    seed = args["seed"]
    wandb.init(
        id = uid,
        resume="allow",
        project="love",
        name=f"env:{name}_seed:{seed}_finetune_goal_lower_right1_wiht_options only",
        group=f"env:{name}_finetune_goal_lower_right",
        sync_tensorboard=False,
        settings=wandb.Settings(start_method="fork"),
    )

    tb_writer = dqn_utils.EpisodeAndStepWriter(
            os.path.join(exp_dir, "tensorboard"))
    wandb_writer = dqn_utils.EpisodeAndStepWriter_wandb(None)
    hssm = torch.load(config.get("checkpoint")).cpu()
    hssm._use_min_length_boundary_mask = True
    hssm.eval()
    

    if config.get("env") == "compile":
        env = grid.ComPILEEnv(
            1, sparse_reward=config.get("sparse_reward"),
            visit_length=config.get("visit_length"))
        train_loader = utils.compile_loader(100)[0]
        hssm.post_obs_state._output_normal = True
        hssm._output_normal = True
    # elif config.get("env") == "3d":
    #     env = world3d.MultiTask3DEnv(
    #             seed=1, num_objects=4, visit_length=4, max_episode_steps=75,
    #             sparse_reward=config.get("sparse_reward"))
    #     env = world3d.PanoramaObservationWrapper(env)
    #     train_loader = utils.miniworld_loader(100)[0]
    #     hssm.post_obs_state._output_normal = False
    #     hssm._output_normal = False
    else:
        env = gym.make(config.get("env"))
        # Fix the initial state
        init_state = (np.array([2.90749422 , 4.92641686 ]), np.array([ 0.00, 0.00]))
        # init_state = (np.array([0.96808476, 6.07712179]), np.array([ 0.00, 0.00]))
        state = mujoco_py.cymj.MjSimState(time=0.0,
                                        qpos=init_state[0], qvel=init_state[1], act=None, udd_state={})
        env.sim.set_state(state)
        train_loader = utils.d4rl_loader(100, config.get("env"))[0]
        hssm.post_obs_state._output_normal = True
        hssm._output_normal = True

    if config.get("oracle", False):
        assert config.get("env") == "compile"
        env = option_wrapper.OracleOptionWrapper(env)
    else:
        if "maze" in config.get("env"):
            env = option_wrapper.OptionWrapperContinous(
                env, hssm, train_loader, train_loader.dataset.seq_size, 1,
                threshold=config.get("threshold"),
                recurrent=config.get("recurrent"))
        else:
            env = option_wrapper.OptionWrapper(
                    env, hssm, train_loader, train_loader.dataset.seq_size, 1,
                    threshold=config.get("threshold"),
                    recurrent=config.get("recurrent"))
        hssm = hssm.cuda()

    # Use GPU if possible
    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device("cuda:0")

    print("Device: {}".format(device))

    agent = dqn.DQNAgent.from_config(config.get("agent"), env)
    # best_option(config.get("env"), 10, hssm, 10)
    # quit()

    # Behavior Cloning
    if config.get("bc"):
        if config.get("env") == "compile":
          agent.behavioral_clone(
                  np.load("compile.npy", allow_pickle=True), 0,
                  num_epochs=50)
        elif config.get("env") == "3d":
          with open("world3d.pkl", "rb") as f:
              trajectories = pickle.load(f)
          interleaved_trajectories = []
          for states, actions in zip(
                  trajectories["states"], trajectories["actions"]):
              episode = []
              for state, action, next_state in zip(states, actions, states[1:]):
                  episode.append((state[0], action, next_state[0]))
              interleaved_trajectories.append(episode)
          agent.behavioral_clone(
              np.array(interleaved_trajectories), 0, num_epochs=50)

    total_steps = 0
    train_rewards = collections.deque(maxlen=100)
    test_rewards = collections.deque(maxlen=100)
    visualize_dir = os.path.join(exp_dir, "visualize")
    os.makedirs(visualize_dir, exist_ok=False)
    for episode_num in tqdm.tqdm(range(150000)):
        episode= run_episode(
            env, agent, experience_observers=[agent.update], return_render=False)[0]

        total_steps += sum(exp.info.get("steps", 1) for exp in episode)
        train_rewards.append(sum(exp.reward for exp in episode))

        if episode_num % 1 == 0:
            return_render = episode_num % 100 == 0
            episode, render = run_episode(
                    env, agent, test=True, return_render=False)
            test_rewards.append(sum(exp.reward for exp in episode))
            # if True:
            #     frames = [frame.image() for frame in render]
            #     episodic_returns = sum(exp.reward for exp in episode)
            #     save_path = os.path.join(visualize_dir, f"{episode_num}.gif")
            #     frames[0].save(save_path, save_all=True, append_images=frames[1:],
            #                    duration=750, loop=0, optimize=True, quality=20)

        
        if episode_num % 10 == 0:
            print("Here")
            tb_writer.add_scalar(
                    "reward/train", np.mean(train_rewards), episode_num,
                    total_steps)
            wandb_writer.add_scalar(
                    "reward/train", np.mean(train_rewards), episode_num,
                    total_steps)

            tb_writer.add_scalar(
                    "reward/test", np.mean(test_rewards), episode_num,
                    total_steps)
            wandb_writer.add_scalar(
                    "reward/test", np.mean(test_rewards), episode_num,
                    total_steps)
            
            render = np.transpose(np.array(render),(0,3,1,2))
            wandb.log({'eval/video': wandb.Video(render[::8,:,::2,::2], fps=6,format="gif")}, step=total_steps)
            
            for k, v in agent.stats.items():
                if v is not None:
                    tb_writer.add_scalar(k, v, episode_num, total_steps)
        if total_steps >= 100000:
            break


if __name__ == '__main__':
    main()
