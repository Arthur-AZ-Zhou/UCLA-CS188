import numpy as np
import robosuite as suite
from policies import *

task_policies = {
    "Lift": LiftPolicy,
    "Stack": StackPolicy,
    "Door": DoorPolicy
}

for task, policy_class in task_policies.items():
    print("Testing task: " + task)

    env = suite.make(
        env_name=task,
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    
    obs = env.reset()
    policy = policy_class(obs)
    
    while True:
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)
        
        env.render()
        if reward == 1.0 or done:
            if (reward == 1.0):
                print("TASK SUCCEEDED")
            else: 
                print("TASK FAILED")

            break