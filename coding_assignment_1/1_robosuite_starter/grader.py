import numpy as np
import robosuite as suite
from policies import *

num_trials = 10 # variable
success_rates = {}

test_tasks = {
	"Lift":  LiftPolicy,
	"Stack": StackPolicy,
	"Door": DoorPolicy  
}

for task_name in test_tasks:
    # create environment instance
    env = suite.make(
        env_name=task_name, 
        robots="Panda",        
        #has_renderer=True, # only works on local machine
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False
    )
    success_rates[task_name] = 0

    for _ in range(num_trials):
        # reset the environment
        obs = env.reset()
        policy = test_tasks[task_name](obs)
        
        while True:
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)  # take action in the environment
            #env.render()
            if reward == 1.0:
                success_rates[task_name] += 1.0
                break
            if done: break

    success_rates[task_name] /= num_trials
    env.close()
    print(task_name, success_rates[task_name])

score = sum([success_rates[t]*2.0 for t in success_rates])
print("Score:", score)