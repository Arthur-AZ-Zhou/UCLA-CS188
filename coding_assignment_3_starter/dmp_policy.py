import numpy as np
from collections import defaultdict
from dmp import DMP
from pid import PID


class DMPPolicyWithPID:
    """
    A policy that follows a demonstrated path with DMPs and PID control.

    The demonstration is split into segments based on grasp toggles.  
    The first segment's endpoint is re-targeted to a new object pose.
    Subsequent segments replay the original DMP rollouts.

    Args:
        square_obs (dict): 'SquareNut_pos' observed
        demo_path (str): path to .npz file with demo data.
        dt (float): control timestep.
        n_bfs (int): number of basis functions per DMP.
    """
    def __init__(self, square_pos, demo_path='demonstration_data.npz', dt=0.01, n_bfs=20):
        self.dt = dt
        # Load and parse demo [DO NOT CHANGE]
        raw = np.load(demo_path)
        demos = defaultdict(dict)
        for key in raw.files:
            prefix, trial, field = key.split('_', 2)
            demos[f"{prefix}_{trial}"][field] = raw[key]
        demo = demos['demo_98']

        # Extract trajectories and grasp
        ee_pos = demo['obs_robot0_eef_pos']  # (T,3)
        ee_grasp = demo['actions'][:, -1:].astype(int)  # (T,1)
        segments = self.detect_grasp_segments(ee_grasp)

        # Compute offset for first segment to new object pose
        demo_obj_pos = demo['obs_object'][0, :3]
        new_obj_pos = square_pos
        start, end = segments[0]
        offset = ee_pos[end-1] - demo_obj_pos

        # TODO: Fit DMPs and generate segment trajectories
        raise NotImplementedError
        


    def detect_grasp_segments(self, grasp_flags: np.ndarray) -> list:
        """
        Identify segments based on grasp toggles.

        Args:
            grasp_flags (np.ndarray): (T,1) array of grasp signals.

        Returns:
            List[Tuple[int,int]]: start and end indices per segment.
        """
        # TODO: implement boundary detection
        raise NotImplementedError




    def get_action(self, robot_eef_pos: np.ndarray) -> np.ndarray:
        """
        Compute next action for the robot's end-effector.

        Args:
            robot_eef_pos (np.ndarray): Current end-effector position [x,y,z].

        Returns:
            np.ndarray: Action vector [dx,dy,dz,0,0,0,grasp].
        """
        # TODO: select current segment and step
        # TODO: compute PID-based delta_pos
        # TODO: assemble action (zero rotation + grasp)
        raise NotImplementedError
