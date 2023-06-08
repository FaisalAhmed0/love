import os
import time

import numpy as np
from train_rl import main
from cluster import exit_for_resume, read_params_from_cmdline, save_metrics_params



if __name__ == "__main__":
    # Error before update_params (has separate handling)
    if np.random.rand() < 0.05:
        raise ValueError("5 percent of all jobs die early for testing")
    

    params = read_params_from_cmdline()
    main(params=params)
    # simulate that the jobs take some time
    # max_sleep_time = params.get("max_sleep_time", 10)
    # time.sleep(np.random.randint(0, max_sleep_time))

    # result_file = os.path.join(params.working_dir, "result.npy")
    # os.makedirs(params.working_dir, exist_ok=True)
    # # here we do a little simulation for checkpointing and resuming
    # if os.path.isfile(result_file):
    #     # If there is a result to resume
    #     noiseless_result = np.load(result_file)
    # else:
    #     # Otherwise compute result, checkpoint it and exit
    #     noiseless_result = fn_to_optimize(**params.fn_args)
    #     print(f"save result to {result_file}")
    #     np.save(result_file, noiseless_result)
    #     if "test_resume" in params and params.test_resume:
    #         exit_for_resume()

    # noisy_result = noiseless_result + 0.5 * np.random.normal()
    # metrics = {"result": noisy_result, "noiseless_result": noiseless_result}
    # save_metrics_params(metrics, params)
    # print(noiseless_result)
