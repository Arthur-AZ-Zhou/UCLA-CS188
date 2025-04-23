import numpy as np

"""
DO NOT MODIFY THIS FILE
"""


class PID:
    def __init__(self, kp, ki, kd, target):
        """
        Initialize a variable-dimension PID controller.

        Args:
            kp (float or list): Proportional gain(s) per axis (or scalar).
            ki (float or list): Integral gain(s) per axis (or scalar).
            kd (float or list): Derivative gain(s) per axis (or scalar).
            target (tuple or array): Target position of any dimension.
        """
        self.setpoint = np.array(target, dtype=np.float64)
        dim = self.setpoint.shape[0]

        self.kp = np.array(kp if hasattr(kp, '__len__') else [kp] * dim, dtype=np.float64)
        self.ki = np.array(ki if hasattr(ki, '__len__') else [ki] * dim, dtype=np.float64)
        self.kd = np.array(kd if hasattr(kd, '__len__') else [kd] * dim, dtype=np.float64)

        self._prev_error = np.zeros(dim)
        self._integral = np.zeros(dim)

    def reset(self, target=None):
        """
        Reset the internal state of the PID controller.

        Args:
            target (optional): New target to reset to.
        """
        if target is not None:
            self.setpoint = np.array(target, dtype=np.float64)
            dim = self.setpoint.shape[0]
            self._prev_error = np.zeros(dim)
            self._integral = np.zeros(dim)
        else:
            self._prev_error.fill(0)
            self._integral.fill(0)

    def get_error(self):
        """
        Returns:
            float: Magnitude of the last error vector.
        """
        return np.linalg.norm(self._prev_error)

    def update(self, current_pos, dt):
        """
        Compute the PID control signal.

        Args:
            current_pos (array-like): Current position (any dimension).
            dt (float): Time delta in seconds.

        Returns:
            np.ndarray: Control output vector.
        """
        current_pos = np.array(current_pos, dtype=np.float64)
        error = self.setpoint - current_pos

        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else np.zeros_like(error)

        output = (
            self.kp * error +
            self.ki * self._integral +
            self.kd * derivative
        )

        self._prev_error = error
        return output


class MoveToPolicy(object):
     
    def __init__(self, obs):
        """
        Initialize the LiftPolicy with the first observation from the environment.

        Args:
            obs (dict): Initial observation from the environment. Must include:
                - 'cube_pos': The position of the cube to be lifted.
        """
        target_pos = obs
        self.dt = 0.05
        # Initialize PID controller to first target (above cube)
        self.pid = PID(1, 2, 0.5, target=target_pos)
        

    def get_action(self, robot0_eef_pos):

        action = np.zeros(7)

        action[-1] = 1   # close gripper
        current_pos = robot0_eef_pos 
        action[:3] = self.pid.update(current_pos, self.dt)

        return action



class LiftPolicy(object):
    """
    A simple PID-based policy for a robotic arm to lift an object in three phases:
    1. Move above the object.
    2. Lower to grasp the object.
    3. Lift the object again.

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
        target_pos = obs
        self.offset = np.array([0, 0.0, 0.1])  # Offset above the cube
        self.dt = 0.1  # Time step for PID update
        self.timeout = 100 # steps per phase
        self.phase_steps = 0

        # Initialize PID controller to first target (above cube)
        self.pid = PID(2, 1, 1, target=target_pos + self.offset)

        self.phase = 0  # 0: approach, 1: descend, 2: lift
        self.target_pos = [
            target_pos + self.offset,  # phase 0: above
            target_pos,                # phase 1: on cube
            target_pos-0.005,                # phase 1: on cube
            target_pos + self.offset  # phase 2: lift
        ]

    def get_action_proprio(self, robot0_eef_pos):
        """
        Compute the next action for the robot based on current proprioception eef_pos observation.

        Args:
            obs (dict): Current observation. Must include:
                - 'robot0_eef_pos': Current end-effector position.
                - 'cube_pos': Current position of the cube.

        Returns:
            np.ndarray: 7D action array for robosuite:
                - action[:3]: XYZ end-effector velocity (from PID)
                - action[-1]: Gripper command (1 to close, -1 to open)
        """
        action = np.zeros(7)
        self.phase_steps += 1

        # Gripper control
        if self.phase >= 2 :
            action[-1] = 1   # close gripper
        else:
            action[-1] = -1  # open gripper

        # Positional PID control
        #print(obs.keys())
        current_pos = robot0_eef_pos 
        action[:3] = self.pid.update(current_pos, self.dt)

        # Check if close enough to target, move to next phase
        err = self.pid.get_error()
        if (err < 0.0045 or self.phase_steps > self.timeout) and self.phase < 3:
            print(self.phase_steps)
            self.phase += 1
            self.phase_steps = 0
            self.pid.reset(target=self.target_pos[self.phase])

        return action


    def get_action_lowdim(self, obs):
        """
        Compute the next action for the robot based on current low-dim observation.

        Args:
            obs (dict): Current observation. Must include:
                - 'robot0_eef_pos': Current end-effector position.
                - 'cube_pos': Current position of the cube.

        Returns:
            np.ndarray: 7D action array for robosuite:
                - action[:3]: XYZ end-effector velocity (from PID)
                - action[-1]: Gripper command (1 to close, -1 to open)
        """
        action = np.zeros(7)
        self.phase_steps += 1

        # Gripper control
        if self.phase >= 2 :
            action[-1] = 1   # close gripper
        else:
            action[-1] = -1  # open gripper

        # Positional PID control
        #print(obs.keys())
        current_pos = obs['robot0_eef_pos']
        action[:3] = self.pid.update(current_pos, self.dt)

        # Check if close enough to target, move to next phase
        err = self.pid.get_error()
        if (err < 0.0045 or self.phase_steps > self.timeout) and self.phase < 3:
            print(self.phase_steps)
            self.phase += 1
            self.phase_steps = 0
            self.pid.reset(target=self.target_pos[self.phase])

        return action


