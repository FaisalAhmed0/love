from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()
import os
import time

import numpy as np
from train_rl import Workspace
from cluster import exit_for_resume, read_params_from_cmdline, save_metrics_params



if __name__ == "__main__":    
    params = read_params_from_cmdline()
    workspace = Workspace(params)
    exitcode = workspace.main()
