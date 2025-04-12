import numpy as np
import robosuite as suite
from policies import *

# Test configuration
tasks = ["Lift", "Stack", "Door"]
policies = [LiftPolicy, StackPolicy, DoorPolicy]

# Run each task
for task, policy_class in zip(tasks, policies):

    print(f"Testing {task} task...")
    
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
            print(f"Task {'succeeded' if reward == 1.0 else 'failed'}")
            break
    
    env.close()