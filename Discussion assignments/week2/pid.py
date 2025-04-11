import numpy as np
import robosuite as suite

# pid.py
class PController:
    def __init__(self, kp, target):
        """
        Initialize a proportional controller.

        Args:
            kp (float): Proportional gain.
            target (tuple or array): Target position.
        """
        self.kp = kp
        self.target = target

    def reset(self, target=None):
        """
        Reset the target position.

        Args:
            target (tuple or array, optional): New target position.
        """
        self.target = target

    def update(self, current_pos):
        """
        Compute the control signal.

        Args:
            current_pos (array-like): Current position.

        Returns:
            np.ndarray: Control output vector.
        """
        currentPosition = np.array(current_pos)
        target = np.array(self.target)
        error = target - currentPosition
        return self.kp * error