import os
import time
from pathlib import Path
import numpy as np
from train_rl import Workspace
from cluster import exit_for_resume, read_params_from_cmdline, save_metrics_params



def main():
    params = read_params_from_cmdline()
    # check if the path already exists
    resume = False
    name = params["name"]
    snapshot_dir = Path(f"/home/fmohamed/love_snapshots_{name}")
    snapshot = snapshot_dir / f'snapshot_latest.pt'
    print(f"snapshot_dir:{snapshot}")
    if os.path.exists(snapshot):
        resume = True
        workspace = Workspace(params, resume)
        workspace.load_snapshot()
        print("This path exists")
    else:
        workspace = Workspace(params, resume)
    exitcode = workspace.main()
    if exitcode == 3:
        return exit_for_resume()
    else:
        return 0


if __name__ == "__main__":    
    main()
    

