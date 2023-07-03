from runner_bilevel import Runner_Bilevel, Runner_Stochastic, Runner_C_Bilevel
from common.arguments import get_args
from common.utils import make_highway_env
import numpy as np
import json
import os


if __name__ == '__main__':
    # get the params
    args = get_args()

    # set train params
    # args.file_path = "./merge_env_result/exp1"
    # args.file_path = "./roundabout_env_result/exp1"
    # args.file_path = "./intersection_env_result/exp1"
    # args.file_path = "./racetrack_env_result/exp1"

    seed = [0,1,2]
    for i in seed:
        args.seed = i
        args.save_dir = args.file_path + "/seed_" + str(args.seed)
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        with open(args.file_path+'/config.json','r') as f:
            vars(args).update(json.load(f))
        
        # set env
        env, eval_env, args = make_highway_env(args)

        np.random.seed(args.seed)

        # choose action type and algorithm
        if args.action_type == "continuous":
            # unconstrained stackelberg maddpg
            if args.version == "bilevel":
                runner = Runner_Bilevel(args, env, eval_env)
            # constrained stackelberg maddpg
            elif args.version == "c_bilevel":
                runner = Runner_C_Bilevel(args, env, eval_env)
        elif args.action_type == "discrete":
            # constrained or unconstrained(by setting extreme high cost threshold) stackelberg Q learning
            runner = Runner_Stochastic(args, env, eval_env)

        # train or evaluate
        if args.evaluate:
            returns = runner.evaluate()
            print('Average returns is', returns)
        else:
            runner.run()

        # record video
        if args.record_video:
            runner.record_video()

        

