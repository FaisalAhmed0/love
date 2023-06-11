import os
import time
from pathlib import Path
import numpy as np
from train_rl import Workspace
from main import main
from cluster import exit_for_resume, read_params_from_cmdline, save_metrics_params
import json

# os.environ['PYTHONPATH'] = '/home/fmohamed/love'


def main_cluster():
    with open("/home/fmohamed/love/cluster_settings/cfg_finetune.json") as f:
        all_parma = json.load(f)
    print(f"all_parma:{all_parma}")
    params = read_params_from_cmdline()
    params["config_bindings"] = all_parma["config_bindings"]
    print(f"params:{params}")
    exitcode = main(params)
    if exitcode == 3:
        return exit_for_resume()
    else:
        return 0


if __name__ == "__main__":    
    main_cluster()
    

