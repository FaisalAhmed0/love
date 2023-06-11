import os
import time
from pathlib import Path
import numpy as np
from train_rl import Workspace
from dqn import main
from cluster import exit_for_resume, read_params_from_cmdline, save_metrics_params

os.environ['PYTHONPATH'] = '/home/fmohamed/love'


def main_cluster():
    params = read_params_from_cmdline()
    print(f"params:{params}")
    exitcode = main(params)
    if exitcode == 3:
        return exit_for_resume()
    else:
        return 0


if __name__ == "__main__":    
    main_cluster()
    

