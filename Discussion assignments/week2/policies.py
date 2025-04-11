import numpy as np
import robosuite as suite

# policies.py    
class HoverPolicy(object):
    """
    A simple P-controller policy for a robotic arm to move above an object in two phases:
    1. (Propose phase one here)
    2. (Propose phase two here)
    We only need a proportional controller to drive the robot's end-effector!
    """
    def __init__(self, obs):
        """
        Initialize the HoverPolicy with the first observation from the environment.
        Args:
            obs (dict): Initial observation from the environment. Must include:
                - 'cube_pos': The position of the cube to be touched.
        """
        self.cube_pos = obs["cube_pos"]
        

    def get_action(self, obs):
        """
        Compute the next action for the robot based on current observation.

        Args:
            obs (dict): Current observation. Must include:
                - 'robot0_eef_pos': Current end-effector position.
                - 'cube_pos': Current position of the cube.

        Returns:
            np.ndarray: 7D action array for robosuite OSC:
                - action[-1]: Gripper command (1 to close, -1 to open)
        """
        # How many dimensions are these?
        eef_pos = obs["robot0_eef_pos"]
        current_cube_pos = obs["cube_pos"]

        # State logic
        # Implement here!
        # End of state logic

        # Use pcontroller method to update control signal.
        ctrl_output = None # Implement this!

        # Q: for students: What is the action space of the robot?
        action = np.zeros(7)

        # Q: for students: Why do we only set the first 3 values?
        # What do the first 3 values represent??
        action[0:3] = ctrl_output

        return action