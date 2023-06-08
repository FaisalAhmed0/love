import os
import time

import numpy as np
from train_rl import main
from cluster import exit_for_resume, read_params_from_cmdline, save_metrics_params



if __name__ == "__main__":    

    params = read_params_from_cmdline()
    main(params=params)
