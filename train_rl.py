# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(300, 300))
# display.start()
import wandb
import time
from datetime import datetime
from modules import (
    GridDecoder,
    GridActionEncoder,
)
import modules
import utils
from hssm_rl import EnvModel
from pathlib import Path
# from world3d import world3d
# from gym_miniworld import miniworld
from grid_world import grid
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np
import logging
import os
import sys
import argparse



LOGGER = logging.getLogger(__name__)
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def parse_args():

    defualt_params = {
        "wandb": True,
        "seed": 1,
        "name": "st",

        # data size
        "dataset_path": "./data/demos",
        "batch_size": 128,
        "seq-size": 6,
        "init_size": 1,

        # model size
        "state_size": 8,
        "belief_size": 128,
        "num_layers": 5,
        "latent_n": 10,

        # observation distribution
        "obs_std": 1.0,
        "obs_bit": 5,

        # optimization
        "learn_rate": 0.0005,
        "grad_clip": 10.0,
        "max_iters": 100000,

        # subsequence prior params
        "seg_num": 100,
        "seg_len": 100,

        # gumbel params
        "max_beta": 1.0,
        "min_beta": 0.1,
        "beta_anneal": 100,

        # log dir
        "log_dir": "./asset/log/",

        # coding length params
        "kl_coeff": 1.0,
        "rec_coeff": 1.0,
        "use_abs_pos_kl": 0,
        "coding_len_coeff": 1.0,
        "use_min_length_boundary_mask": False,

        "action_type": "d",  # d for discrete, c for continuous


        # baselines
        "ddo": False,

        # max runtime in the cluster
        "max_runtime": 30
    }
    return defualt_params
    # parser = argparse.ArgumentParser(description="vta agr parser")
    # parser.add_argument("--wandb", action="store_true")
    # parser.add_argument("--seed", type=int, default=1)
    # parser.add_argument("--name", type=str, default="st")

    # # data size
    # parser.add_argument("--dataset-path", type=str, default="./data/demos")
    # parser.add_argument("--batch-size", type=int, default=128)
    # parser.add_argument("--seq-size", type=int, default=6)
    # parser.add_argument("--init-size", type=int, default=1)

    # # model size
    # parser.add_argument("--state-size", type=int, default=8)
    # parser.add_argument("--belief-size", type=int, default=128)
    # parser.add_argument("--num-layers", type=int, default=5)
    # parser.add_argument("--latent-n", type=int, default=10)

    # # observation distribution
    # parser.add_argument("--obs-std", type=float, default=1.0)
    # parser.add_argument("--obs-bit", type=int, default=5)

    # # optimization
    # parser.add_argument("--learn-rate", type=float, default=0.0005)
    # parser.add_argument("--grad-clip", type=float, default=10.0)
    # parser.add_argument("--max-iters", type=int, default=100000)

    # # subsequence prior params
    # parser.add_argument("--seg-num", type=int, default=100)
    # parser.add_argument("--seg-len", type=int, default=100)

    # # gumbel params
    # parser.add_argument("--max-beta", type=float, default=1.0)
    # parser.add_argument("--min-beta", type=float, default=0.1)
    # parser.add_argument("--beta-anneal", type=float, default=100)

    # # log dir
    # parser.add_argument("--log-dir", type=str, default="./asset/log/")

    # # coding length params
    # parser.add_argument("--kl_coeff", type=float, default=1.0)
    # parser.add_argument("--rec_coeff", type=float, default=1.0)
    # parser.add_argument("--use_abs_pos_kl", type=float, default=0)
    # parser.add_argument("--coding_len_coeff", type=float, default=1.0)
    # parser.add_argument("--use_min_length_boundary_mask", action="store_true")

    # # baselines
    # parser.add_argument("--ddo", action="store_true")
    # return parser.parse_args()


def date_str():
    s = str(datetime.now())
    d, t = s.split(" ")
    t = "-".join(t.split(":")[:-1])
    return d + "-" + t


def set_exp_name(args):
    exp_name = args["name"] + "__seed__" + str(args['seed']) + "_" + date_str()
    return exp_name


class Workspace:
    def __init__(self, params, resume):
        
        # parse arguments
        self.args = parse_args()
        self.cmd_args = params
        if self.cmd_args:
            for key in self.cmd_args:
                self.args[key] = self.cmd_args[key]
        self.params = self.args
        if not self.args["wandb"]:
            os.environ["WANDB_MODE"] = "offline"

        # fix seed
        np.random.seed(self.args["seed"])
        torch.manual_seed(self.args["seed"])
        torch.cuda.manual_seed_all(self.args["seed"])
        torch.backends.cudnn.deterministic = True

        # set logger
        log_format = "[%(asctime)s] %(message)s"
        logging.basicConfig(level=logging.INFO,
                            format=log_format, stream=sys.stderr)

        # set size
        self.init_size = self.args["init_size"]

        # set device as gpu
        self.device = torch.device("cuda", 0)
        # device = torch.device("cpu")

        # set writer
        self.exp_name = set_exp_name(self.args)


        # generate an id to resume
        if resume:
            pass
        else:
            self.uid = wandb.util.generate_id()
            wandb.init(
                id = self.uid,
                resume="allow",
                project="love",
                config=self.args,
                name=self.exp_name,
                group=self.args["name"],
                sync_tensorboard=False,
                settings=wandb.Settings(start_method="fork"),
            )

        LOGGER.info("EXP NAME: " + self.exp_name)
        LOGGER.info(">" * 80)
        LOGGER.info(self.args)
        LOGGER.info(">" * 80)

        # load dataset
        if "compile" in self.args["dataset_path"]:
            self.train_loader, self.test_loader = utils.compile_loader(
                self.args["batch_size"])
            # action encdoer given the trajectory
            self.action_encoder = GridActionEncoder(
                action_size=self.train_loader.dataset.action_size,
                embedding_size=self.args["belief_size"],
            )
            # observation encdoer given the trajectory
            self.encoder = modules.CompILEGridEncoder(
                feat_size=self.args["belief_size"])
            # observation decoder for recounstructing the observation given the state abstraction
            self.decoder = GridDecoder(
                input_size=self.args["belief_size"],
                action_size=self.train_loader.dataset.action_size,
                feat_size=self.args["belief_size"],
            )
            self.output_normal = True
        elif "miniworld" in self.args["dataset_path"]:
            self.train_loader, self.test_loader = utils.miniworld_loader(
                self.args["batch_size"])
            self.action_encoder = GridActionEncoder(
                action_size=self.train_loader.dataset.action_size,
                embedding_size=self.args["belief_size"],
            )
            self.encoder = modules.MiniWorldEncoderPano(input_dim=3)
            self.decoder = GridDecoder(
                input_size=self.args["belief_size"],
                action_size=self.train_loader.dataset.action_size,
                feat_size=self.args["belief_size"],
            )
            self.output_normal = False

        elif "d4rl" in self.args["dataset_path"]:
            self.train_loader, self.test_loader = utils.d4rl_loader(
                self.args["batch_size"], self.args["name"])
            self.action_encoder = modules.D4RLActionEncoder(
                action_size=self.train_loader.dataset.action_size,
                embedding_size=self.args["belief_size"],
            )
            self.encoder = modules.D4RlEncoder()
            self.decoder = GridDecoder(
                input_size=self.args["belief_size"],
                action_size=self.train_loader.dataset.action_size,
                feat_size=self.args["belief_size"],
            )
            self.output_normal = True

        else:
            path = self.args["dataset_path"]
            raise ValueError(f"Unrecognize dataset_path {path}")

        self.seq_size = self.train_loader.dataset.seq_size

        # init models
        use_abs_pos_kl = self.args["use_abs_pos_kl"] == 1.0
        self.model = EnvModel(
            action_encoder=self.action_encoder,
            encoder=self.encoder,
            decoder=self.decoder,
            belief_size=self.args["belief_size"],
            state_size=self.args["state_size"],
            num_layers=self.args["num_layers"],
            max_seg_len=self.args["seg_len"],
            max_seg_num=self.args["seg_num"],
            latent_n=self.args["latent_n"],
            kl_coeff=self.args["kl_coeff"],
            rec_coeff=self.args["rec_coeff"],
            use_abs_pos_kl=use_abs_pos_kl,
            coding_len_coeff=self.args["coding_len_coeff"],
            use_min_length_boundary_mask=self.args["use_min_length_boundary_mask"],
            ddo=self.args["ddo"],
            output_normal=self.output_normal,
            action_type=self.args["action_type"]
        ).to(self.device)
        LOGGER.info("Model initialized")

        # init optimizer
        self.optimizer = Adam(params=self.model.parameters(),
                              lr=self.args["learn_rate"], amsgrad=True)

        # test data
        self.pre_test_full_state_list, self.pre_test_full_action_list = next(
            iter(self.test_loader))
        self.pre_test_full_state_list = self.pre_test_full_state_list.to(
            self.device)
        self.pre_test_full_action_list = self.pre_test_full_action_list.to(
            self.device)

        # for each iter
        torch.autograd.set_detect_anomaly(False)
        self.b_idx = 0

    def main(self):
        start_time = time.time()
        b_idx = self.b_idx
        while b_idx <= self.args["max_iters"]:
            for train_obs_list, train_action_list in self.train_loader:
                b_idx += 1
                # mask temp annealing
                if self.args["beta_anneal"]:
                    self.model.state_model.mask_beta = (
                        self.args["max_beta"] - self.args["min_beta"]
                    ) * 0.999 ** (b_idx / self.args["beta_anneal"]) + self.args["beta_anneal"]
                else:
                    self.model.state_model.mask_beta = self.args["max_beta"]

                ##############
                # train time #
                ##############
                train_obs_list = train_obs_list.to(self.device)
                train_action_list = train_action_list.to(self.device)

                # run model with train mode
                self.model.train()
                self.optimizer.zero_grad()
                results = self.model(
                    train_obs_list, train_action_list, self.seq_size, self.init_size, self.args[
                        "obs_std"]
                )

                if self.args["coding_len_coeff"] > 0:
                    if results["obs_cost"].mean() < 0.05:
                        self.model.coding_len_coeff += 0.00002
                    elif b_idx > 0:
                        self.model.coding_len_coeff -= 0.00002

                    self.model.coding_len_coeff = min(
                        0.05, self.model.coding_len_coeff)
                    self.model.coding_len_coeff = max(
                        0.000000, self.model.coding_len_coeff)
                    results["coding_len_coeff"] = self.model.coding_len_coeff

                # get train loss and backward update
                train_total_loss = results["train_loss"]
                train_total_loss.backward()
                if self.args["grad_clip"] > 0.0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args["grad_clip"], error_if_nonfinite=True)
                self.optimizer.step()

                # log
                if b_idx % 5 == 0:
                    base_path = "/home/fmohamed/"
                    exp_dir = Path(os.path.join(f"{base_path}/experiments", f"{name}_seed_{seed}"))
                    exp_dir.mkdir(exist_ok=True, parents=True)
                    utils.record_options(self.args["name"], self.model, self.args["latent_n"], exp_dir)
                    results["grad_norm"] = grad_norm
                    train_stats, log_str, log_data = utils.log_train(
                        results, None, b_idx)
                    if not "d4rl" in self.params["dataset_path"]:
                        # Boundaries for grid world
                        true_boundaries = train_action_list[:,
                                                            self.init_size:-self.init_size] == 4
                        true_boundaries = torch.roll(true_boundaries, 1, -1)
                        true_boundaries[:, 0] = True
                        correct_boundaries = torch.logical_and(
                            results["mask_data"].squeeze(
                                -1) == true_boundaries, true_boundaries
                        ).sum()
                        num_pred_boundaries = results["mask_data"].sum()
                        num_true_boundaries = true_boundaries.sum()
                        train_stats["train/precision"] = (
                            correct_boundaries / num_pred_boundaries
                        )
                        train_stats["train/recall"] = correct_boundaries / \
                            num_true_boundaries

                        LOGGER.info(log_str, *log_data)
                    wandb.log(train_stats, step=b_idx)

                np.set_printoptions(threshold=100000)
                torch.set_printoptions(threshold=100000)

                if b_idx % 200 == 0:
                    exp_dir = os.path.join(
                        "/home/fmohamed/love_experiments", self.args["name"], str(b_idx))
                    os.makedirs(exp_dir, exist_ok=True)
                    for batch_idx in range(min(train_obs_list.shape[0], 10)):
                        states = train_obs_list[batch_idx][self.init_size:-self.init_size]
                        actions = train_action_list[batch_idx][self.init_size:-self.init_size]
                        reconstructed_actions = torch.argmax(results["rec_data"], -1)[
                            batch_idx
                        ]
                        options = results["option_list"][batch_idx]
                        boundaries = results["mask_data"][batch_idx]
                        frames = []
                        curr_option = options[0]
                        if not "d4rl" in self.params["dataset_path"]:
                            for seq_idx in range(states.shape[0]):
                                # read new option if boundary is 1
                                if boundaries[seq_idx].item() == 1:
                                    curr_option = options[seq_idx]

                                # panorama observation for miniworld
                                if self.args["dataset_path"] == "miniworld":
                                    pass
                                    ################################################
                                    #  Begin of miniworld specific
                                    ################################################
                                    # f = []
                                    # for i in range(5):
                                    #     f.append(
                                    #         states[seq_idx][:, :, i * 3: (i + 1) * 3])
                                    # frame = torch.cat(f[::-1], axis=1)
                                    # frame = world3d.Render(
                                    #     frame.cpu().data.numpy())
                                    # frame.write_text(
                                    #     f"Action: {repr(miniworld.MiniWorldEnv.Actions(actions[seq_idx].item()))}")
                                    # frame.write_text(
                                    #     f"Reconstructed: {repr(miniworld.MiniWorldEnv.Actions(reconstructed_actions[seq_idx].item()))}")
                                    # if (
                                    #     actions[seq_idx].item()
                                    #     == reconstructed_actions[seq_idx].item()
                                    # ):
                                    #     frame.write_text("CORRECT")
                                    # else:
                                    #     frame.write_text("WRONG")

                                    # if actions[seq_idx].item() == miniworld.MiniWorldEnv.Actions.pickup:
                                    #     frame.write_text("PICKUP")
                                    # else:
                                    #     frame.write_text("NOT PICKUP")

                                    ################################################
                                    #  End of miniworld specific
                                    ################################################
                                elif self.args["dataset_path"] == "compile":
                                    ################################################
                                    #  Begin of compile specific
                                    ################################################
                                    frame = grid.GridRender(10, 10)
                                    # this double for loop is for visualization
                                    for x in range(10):
                                        for y in range(10):
                                            obj = np.argmax(
                                                states[seq_idx][x][y].cpu(
                                                ).data.numpy()
                                            )
                                            if (
                                                obj == grid.ComPILEObject.num_types()
                                                or states[seq_idx][x][y][
                                                    grid.ComPILEObject.num_types()
                                                ]
                                            ):
                                                frame.draw_rectangle(
                                                    np.array(
                                                        (x, y)), 0.9, "cyan"
                                                )
                                            elif obj == grid.ComPILEObject.num_types() + 1:
                                                frame.draw_rectangle(
                                                    np.array(
                                                        (x, y)), 0.7, "black"
                                                )
                                            elif states[seq_idx][x][y][obj] == 1:
                                                frame.draw_rectangle(
                                                    np.array((x, y)),
                                                    0.4,
                                                    grid.ComPILEObject.COLORS[obj],
                                                )
                                    frame.write_text(
                                        f"Action: {repr(grid.Action(actions[seq_idx].item()))}"
                                    )
                                    frame.write_text(
                                        f"Reconstructed: {repr(grid.Action(reconstructed_actions[seq_idx].item()))}"
                                    )
                                    if (
                                        actions[seq_idx].item()
                                        == reconstructed_actions[seq_idx].item()
                                    ):
                                        frame.write_text("CORRECT")
                                    else:
                                        frame.write_text("WRONG")
                                    ################################################
                                    #  End of compile specific
                                    ################################################
                                frame.write_text(f"Option: {curr_option}")
                                frame.write_text(
                                    f"Boundary: {boundaries[seq_idx].item()}")
                                frame.write_text(
                                    f"Obs NLL: {results['obs_cost'].mean()}")
                                frame.write_text(
                                    f"Coding length: {results['encoding_length'].item()}"
                                )
                                frame.write_text(
                                    f"Num reads: {results['mask_data'].sum(1).mean().item()}"
                                )
                                frames.append(frame.image())

                            save_path = os.path.join(
                                exp_dir, f"{batch_idx}.gif")
                            frames[0].save(
                                save_path,
                                save_all=True,
                                append_images=frames[1:],
                                duration=750,
                                loop=0,
                                optimize=True,
                                quality=20,
                            )

                if b_idx % 100 == 0:
                    LOGGER.info("#" * 80)
                    LOGGER.info(">>> option list")
                    LOGGER.info("\n" + repr(results["option_list"][:10]))
                    LOGGER.info(">>> boundary mask list")
                    LOGGER.info(
                        "\n" + repr(results["mask_data"][:10].squeeze(-1)))
                    LOGGER.info(">>> train_action_list")
                    LOGGER.info("\n" + repr(train_action_list[:10]))
                    LOGGER.info(">>> argmax reconstruction")
                    LOGGER.info(
                        "\n" + repr(torch.argmax(results["rec_data"], -1)[:10]))
                    LOGGER.info(">>> diff")
                    if not "d4rl" in self.params["dataset_path"]:
                        LOGGER.info(
                            "\n"
                            + repr(
                                train_action_list[:10, 1:-1]
                                - torch.argmax(results["rec_data"][:10], -1)
                            )
                        )
                    LOGGER.info(">>> marginal")
                    LOGGER.info("\n" + repr(results["marginal"]))
                    LOGGER.info("#" * 80)

                if b_idx % 2000 == 0:
                    name = self.args["name"]
                    seed = self.args["seed"]
                    base_path = "/home/fmohamed/"
                    exp_dir = Path(os.path.join(f"{base_path}/experiments", f"{name}_seed_{seed}"))
                    exp_dir.mkdir(exist_ok=True, parents=True)
                    torch.save(
                        self.model.state_model, os.path.join(
                            str(exp_dir), f"model-{b_idx}.ckpt")
                    )

                #############
                # test time #
                #############
                if b_idx % 100 == 0:
                    with torch.no_grad():
                        ##################
                        # test data elbo #
                        ##################
                        self.model.eval()
                        results = self.model(
                            self.pre_test_full_state_list,
                            self.pre_test_full_action_list,
                            self.seq_size,
                            self.init_size,
                            self.args["obs_std"],
                        )

                        # log
                        test_stats, log_str, log_data = utils.log_test(
                            results, None, b_idx)
                        # Boundaries for grid world
                        if not "d4rl" in self.params["dataset_path"]:
                            true_boundaries = (
                                self.pre_test_full_action_list[:,
                                                            self.init_size:-self.init_size] == 4
                            )
                            true_boundaries = torch.roll(true_boundaries, 1, -1)
                            true_boundaries[:, 0] = True
                            
                            correct_boundaries = torch.logical_and(
                                results["mask_data"].squeeze(
                                    -1) == true_boundaries,
                                true_boundaries,
                            ).sum()
                            num_pred_boundaries = results["mask_data"].sum()
                            num_true_boundaries = true_boundaries.sum()
                            test_stats["valid/precision"] = (
                                correct_boundaries / num_pred_boundaries
                            )
                            test_stats["valid/recall"] = (
                                correct_boundaries / num_true_boundaries
                            )
                            LOGGER.info(log_str, *log_data)
                            wandb.log(test_stats, step=b_idx)
                self.run_time = (time.time() - start_time) / 60
                wandb.log({"train/run_time": self.run_time}, step=b_idx)
                if ((time.time() - start_time) / 60) > self.args["max_runtime"]:
                    self.b_idx = b_idx
                    self.save_snapshot(f"_{b_idx}")
                    print(f"Saving snapshot and exit for resume")
                    return 3

    def save_snapshot(self, suffix):
        # _suffix = suffix
        name = self.args["name"]
        seed = self.args["seed"]
        snapshot_dir = Path(f"/home/fmohamed/love_snapshots_{name}_seed_{seed}")
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_latest.pt'
        # self.last_current_size = self.replay_storage.current_size
        keys_to_save = ['action_encoder', 'args', 'b_idx', 'cmd_args', 'decoder', 'device', 
                        'encoder', 'init_size', 'model', 'optimizer', 'output_normal', 'params',
                        'pre_test_full_action_list', 'pre_test_full_state_list', 'seq_size', 'test_loader', 'train_loader', "uid", "run_time"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
            wandb.save(str(snapshot)) # saves checkpoint to wandb

    def load_snapshot(self):
        print(f"Loading snapshot")
        name = self.args["name"]
        seed = self.args["seed"]
        snapshot_dir = Path(f"/home/fmohamed/love_snapshots_{name}_seed_{seed}")
        snapshot = snapshot_dir / f'snapshot_latest.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        # load attributes from the payload
        self.action_encoder = payload["action_encoder"]
        self.args = payload["args"]
        self.b_idx = payload["b_idx"]
        self.cmd_args = payload["cmd_args"]
        self.decoder = payload["decoder"]
        self.device = payload["device"]
        self.encoder = payload["encoder"]
        self.init_size = payload["init_size"]
        self.model = payload["model"]
        self.optimizer = payload["optimizer"]
        self.params = payload["params"]
        self.pre_test_full_action_list = payload["pre_test_full_action_list"]
        self.pre_test_full_state_list = payload["pre_test_full_state_list"]
        self.seq_size = payload["seq_size"]
        self.test_loader = payload["test_loader"]
        self.train_loader = payload["train_loader"]
        self.output_normal = payload["output_normal"]
        self.uid = payload["uid"]
        self.run_time = payload["run_time"]
        wandb.init(
            id = self.uid,
            resume="must",
            project="love",
            name=self.exp_name,
            sync_tensorboard=False,
            settings=wandb.Settings(start_method="fork"),
        )
        

    


if __name__ == "__main__":
    pass
