import numpy as np
import robosuite as suite
from pid import PID

class LiftPolicy(object):
    """
    A simple PID-based policy for a robotic arm to lift an object in three phases:
    1. Move above the object.
    2. Lower to grasp the object.
    3. Lift the object.

    The policy uses a PID controller to drive the robot's end-effector to a sequence of target positions
    while managing the gripper state based on the current phase of motion.
    """

    def __init__(self, obs):
        """
        Initialize the LiftPolicy with the first observation from the environment.

        Args:
            obs (dict): Initial observation from the environment. Must include:
                - 'cube_pos': The position of the cube to be lifted.
        """
        pass
        
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
        # return action
        pass


class StackPolicy(object):
    pass


class DoorPolicy(object):
    pass