import numpy as np
import robosuite as suite
import time
from pid import PID

class LiftPolicy(object):
    def __init__(self, obs):
        """
        Initialize the LiftPolicy with the first observation from the environment.
        """
        # PID gains for position control (x,y,z)
        # Higher gains for precise positioning
        self.kp_precise = [2.0, 2.0, 2.0]
        self.ki_precise = [0.05, 0.05, 0.05]
        self.kd_precise = [0.2, 0.2, 0.2]
        
        # Lower gains for lifting motion (slower, smoother)
        self.kp_lift = [1.0, 1.0, 0.5]  # Lower z gain for slower lifting
        self.ki_lift = [0.01, 0.01, 0.01]
        self.kd_lift = [0.1, 0.1, 0.1]
        
        cube_pos = obs['cube_pos']
        
        # Define targets
        self.target_above = cube_pos.copy()
        self.target_above[2] += 0.1  # 10cm above cube
        
        self.target_grasp = cube_pos.copy()
        
        self.target_lift = cube_pos.copy()
        self.target_lift[2] += 0.3  # Lift height
        
        # Start with precise positioning
        self.pid = PID(self.kp_precise, self.ki_precise, self.kd_precise, self.target_above)
        self.phase = 0  # 0: move above, 1: lower to grasp, 2: grasping, 3: lift
        self.gripper_open = True
        self.grasp_start_time = None
        self.grasp_duration = 0.5  # 0.5 seconds to ensure grasp
        
    def get_action(self, obs):
        eef_pos = obs['robot0_eef_pos']
        cube_pos = obs['cube_pos']
        error = self.pid.get_error()
        
        # Update targets in case cube moved
        self.target_above = cube_pos.copy()
        self.target_above[2] += 0.1
        self.target_grasp = cube_pos.copy()
        
        # Phase transitions
        if self.phase == 0 and error < 0.02:  # Reached above position
            self.phase = 1
            self.pid.reset(self.target_grasp)
        elif self.phase == 1 and error < 0.01:  # Reached grasp position
            self.phase = 2
            self.gripper_open = False  # Close gripper
            self.grasp_start_time = time.time()  # Start grasp timer
        elif self.phase == 2 and (time.time() - self.grasp_start_time) > self.grasp_duration:
            self.phase = 3
            # Switch to lifting gains and target
            self.pid = PID(self.kp_lift, self.ki_lift, self.kd_lift, self.target_lift)
        
        # Get PID control output
        control = self.pid.update(eef_pos, 0.01)
        
        # Create action vector
        action = np.zeros(7)
        action[:3] = control[:3]
        
        # Gripper control - keep closed during and after grasping phase
        if self.phase >= 2:
            action[-1] = 1  # Keep gripper closed
        else:
            action[-1] = -1  # Open gripper
        
        return action
    
class StackPolicy(object):
    def __init__(self, obs):
        """
        Initialize the StackPolicy with improved grasping and stacking behavior.
        """
        # PID gains for precise positioning
        self.kp_precise = [2.0, 2.0, 2.0]
        self.ki_precise = [0.05, 0.05, 0.05]
        self.kd_precise = [0.2, 0.2, 0.2]
        
        # PID gains for lifting/stacking motions (slower in z-axis)
        self.kp_lift = [1.0, 1.0, 0.5]
        self.ki_lift = [0.01, 0.01, 0.01]
        self.kd_lift = [0.1, 0.1, 0.1]
        
        # Get cube positions
        self.cubeA_pos = obs['cubeA_pos']  # Cube to pick up
        self.cubeB_pos = obs['cubeB_pos']  # Base cube
        
        # Define targets
        self.target_approach_A = self.cubeA_pos.copy()
        self.target_approach_A[2] += 0.1  # 10cm above cube A
        
        self.target_grasp_A = self.cubeA_pos.copy()
        
        self.target_lift = self.cubeA_pos.copy()
        self.target_lift[2] += 0.1  # Lift height
        
        self.target_approach_B = self.cubeB_pos.copy()
        self.target_approach_B[2] += 0.1  # Stacking height
        
        # Initialize PID and state
        self.pid = PID(self.kp_precise, self.ki_precise, self.kd_precise, self.target_approach_A)
        self.phase = 0  # 0: approach A, 1: grasp A, 2: lift A, 3: approach B, 4: release
        self.gripper_open = True
        self.grasp_start_time = None
        self.grasp_duration = 0.5  # 0.5 seconds to ensure grasp
        self.release_start_time = None
        self.release_duration = 1  # 0.3 seconds to ensure release
        self.settle_duration = 0.5
        self.complete = False;
        
    def get_action(self, obs):
        if self.complete:  # NEW: Return neutral action if completed
            return np.zeros(7)
            
        eef_pos = obs['robot0_eef_pos']
        self.cubeA_pos = obs['cubeA_pos']
        self.cubeB_pos = obs['cubeB_pos']
        error = self.pid.get_error()
        
        # Update moving targets
        self.target_approach_A = self.cubeA_pos.copy()
        self.target_approach_A[2] += 0.1
        self.target_grasp_A = self.cubeA_pos.copy()
        
        # Phase transitions
        if self.phase == 0 and error < 0.02:
            self.phase = 1
            self.pid.reset(self.target_grasp_A)
        elif self.phase == 1 and error < 0.01:
            self.phase = 2
            self.gripper_open = False
            self.grasp_start_time = time.time()
        elif self.phase == 2 and (time.time() - self.grasp_start_time) > self.grasp_duration:
            self.phase = 3
            self.pid = PID(self.kp_lift, self.ki_lift, self.kd_lift, self.target_lift)
        elif self.phase == 3 and error < 0.03:
            self.phase = 4
            self.pid = PID(self.kp_precise, self.ki_precise, self.kd_precise, self.target_approach_B)
        elif self.phase == 4 and error < 0.02:
            self.phase = 5
            self.pid.reset(self.cubeB_pos + np.array([0, 0, 0.05]))
            self.release_start_time = time.time()
        elif self.phase == 5:
            if (time.time() - self.release_start_time) > self.release_duration:
                self.gripper_open = True
                self.phase = 6  # NEW: Enter settle phase
                self.settle_start_time = time.time()
        elif self.phase == 6:  # NEW: Settle phase
            if (time.time() - self.settle_start_time) > self.settle_duration:
                self.complete = True  # Mark as fully completed
        
        # Get PID control output
        control = self.pid.update(eef_pos, 0.01) if self.phase < 6 else np.zeros(3)
        
        # Create action vector
        action = np.zeros(7)
        action[:3] = control[:3]
        
        # Gripper control
        if 2 <= self.phase < 5:  # Keep closed until release
            action[-1] = 1
        else:
            action[-1] = -1 if self.gripper_open else 1
        
        return action

    def is_complete(self):  # NEW: Completion check method
        return self.complete

class DoorPolicy(object):
    def __init__(self, obs):
        """
        Robust door opening policy for opposite-side doors (right side of screen).
        Implements:
        1. Correct approach from the front-left side
        2. Firm grasp with downward pressure
        3. Proper downward-and-left turning motion
        """
        # PID gains
        self.kp_approach = [8.0, 8.0, 8.0]  # High precision approach
        self.ki_approach = [0.1, 0.1, 0.1]
        self.kd_approach = [0.5, 0.5, 0.5]
        
        self.kp_turn = [4.0, 4.0, 2.0]  # Smoother turning gains
        self.ki_turn = [0.05, 0.05, 0.0]
        self.kd_turn = [0.4, 0.4, 0.2]

        # Get initial positions
        self.handle_pos = obs['handle_pos']
        self.door_pos = obs['door_pos']
        
        # Calculate proper approach direction for opposite-side door
        self.door_normal = self.handle_pos - self.door_pos
        self.door_normal[2] = 0  # Horizontal only
        self.door_normal = self.door_normal / np.linalg.norm(self.door_normal)
        
        # Approach from front-left (opposite of normal)
        self.approach_dir = -np.cross(self.door_normal, [0, 0, 1])
        self.approach_dir = self.approach_dir / np.linalg.norm(self.approach_dir)

        # Target positions
        self.target_approach = self.handle_pos + self.approach_dir * 0.15  # 15cm offset
        self.target_approach[2] += 0.12  # 12cm above
        
        self.target_pre_grasp = self.handle_pos + self.approach_dir * 0.05  # 5cm offset
        self.target_pre_grasp[2] += 0.05  # 5cm above
        
        # Final grasp pushes into handle and down
        self.target_grasp = self.handle_pos.copy()
        self.target_grasp -= self.approach_dir * 0.03  # Into handle
        self.target_grasp[2] -= 0.02  # Downward
        
        # Turning parameters
        self.turn_angle = -90  # Degrees (downward turn)
        self.turn_radius = 0.12  # Slightly larger radius
        self.turn_duration = 3.5  # Slower turn
        
        # Initialize controller
        self.pid = PID(self.kp_approach, self.ki_approach, self.kd_approach, self.target_approach)
        self.phase = 0  # 0:approach, 1:pre-grasp, 2:grasp, 3:turn
        self.gripper_open = True
        self.grasp_start_time = None
        self.grasp_duration = 2.0  # Longer grasp
        self.turn_start_time = None
        self.initial_grasp_pos = None

    def get_action(self, obs):
        eef_pos = obs['robot0_eef_pos']
        self.handle_pos = obs['handle_pos']  # Update handle position
        error = self.pid.get_error()
        
        action = np.zeros(7)  # 6DOF + gripper
        
        # Phase transitions
        if self.phase == 0 and error < 0.03:
            self.phase = 1
            self.pid.reset(self.target_pre_grasp)
        elif self.phase == 1 and error < 0.02:
            self.phase = 2
            self.pid.reset(self.target_grasp)
        elif self.phase == 2 and error < 0.01:
            self.phase = 3
            self.gripper_open = False
            self.grasp_start_time = time.time()
            self.initial_grasp_pos = eef_pos.copy()
        elif self.phase == 3:
            if not self.turn_start_time and (time.time() - self.grasp_start_time) > self.grasp_duration:
                self.turn_start_time = time.time()
                self.pid = PID(self.kp_turn, self.ki_turn, self.kd_turn, eef_pos)
            
            if self.turn_start_time:
                turn_progress = min((time.time() - self.turn_start_time) / self.turn_duration, 1.0)
                
                # Corrected turning motion for opposite-side door:
                angle = np.radians(self.turn_angle * turn_progress)
                turn_offset = np.array([
                    self.turn_radius * (1 - np.cos(angle)),  # Forward
                    -self.turn_radius * np.sin(angle),       # Leftward (opposite side)
                    -self.turn_radius * np.sin(angle)        # Downward
                ])
                
                target_pos = self.initial_grasp_pos + turn_offset
                control = self.pid.update(target_pos, 0.01)
                action[:3] = control[:3]
        
        # Gripper control
        action[-1] = -1 if self.gripper_open else 1
        
        return action