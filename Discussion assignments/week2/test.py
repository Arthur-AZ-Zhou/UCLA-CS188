import numpy as np
import robosuite as suite
from policies import HoverPolicy

# Create environment instance
env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# Reset the environment
for _ in range(5):
    obs = env.reset()
    policy = HoverPolicy(obs)
    
    while True:
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)  # take action in the environment
        
        env.render()  # render on display
        if reward == 1.0 or done:
            # Ignore reward for now.
            # You should care about it for the actual assignment.
            break