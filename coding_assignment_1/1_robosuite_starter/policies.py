import numpy as np
import robosuite as suite
import time
from pid import PID

class LiftPolicy(object):
    def __init__(self, obs):
        """
        Initialize the LiftPolicy with the first observation from the environment.
        """
        self.kp_precise = [2.0, 2.0, 2.0] #K_ for axes: (x, y, z)
        self.ki_precise = [0.05, 0.05, 0.05]
        self.kd_precise = [0.2, 0.2, 0.2]
        self.kp_lift = [1.0, 1.0, 0.5] #low = slower and smoother lifts, (x, y, z) again
        self.ki_lift = [0.01, 0.01, 0.01]
        self.kd_lift = [0.1, 0.1, 0.1]
        
        cube_pos = obs['cube_pos']
        
        self.target_height = cube_pos.copy() #grabs cube 10cm above ground
        self.target_height[2] += 0.1
        self.target_grasp = cube_pos.copy() #grab cube at exact coords sitting down
        self.target_lift = cube_pos.copy() #how high we lift the cube
        self.target_lift[2] += 0.5  
        
        self.pid_controller = PID(self.kp_precise, self.ki_precise, self.kd_precise, self.target_height) #init new PID controller

        self.phase = 0  #0: arm moves above, 1: lower arm to grab, 2: grab, 3: lift
        self.gripper_open = True
        self.grasp_start_time = 0
        self.grasp_duration = 0.5  # 0.5 seconds to ensure grasp
        
    def get_action(self, obs):
        eef_pos = obs['robot0_eef_pos'] #end-effector position
        current_cube_pos = obs['cube_pos']
        error = self.pid_controller.get_error() #can think of this as distance to target
        
        if (self.phase == 0 and error < 0.02):  #reached above
            self.phase = 1
            self.pid_controller.reset(self.target_grasp)

        elif (self.phase == 1 and error < 0.01):  #reached grab
            self.phase = 2
            self.gripper_open = False
            self.grasp_start_time = time.time()  

        elif (self.phase == 2 and self.grasp_duration < (time.time() - self.grasp_start_time)):
            self.phase = 3
            self.pid_controller = PID(self.kp_lift, self.ki_lift, self.kd_lift, self.target_lift)
        
        control = self.pid_controller.update(eef_pos, 0.01)
        
        action = np.zeros(7)
        action[:3] = control[:3]
        
        if self.phase >= 2:
            action[-1] = 1 #closed gripper
        else:
            action[-1] = -1 #opened gripper
        
        return action
    
class StackPolicy(object):
    def __init__(self, obs):
        """
        Initialize the StackPolicy with improved grasping and stacking behavior.
        """
        self.kp_precise = [2.0, 2.0, 2.0]
        self.ki_precise = [0.05, 0.05, 0.05]
        self.kd_precise = [0.2, 0.2, 0.2]
        self.kp_lift = [1.0, 1.0, 0.5]
        self.ki_lift = [0.01, 0.01, 0.01]
        self.kd_lift = [0.1, 0.1, 0.1]
        
        self.cube_pos_1 = obs['cubeA_pos'] #red
        self.cube_pos_2 = obs['cubeB_pos'] #green
        
        self.target_height_1 = self.cube_pos_1.copy()
        self.target_height_1[2] += 0.1  #grabs red cube 10cm off ground
        self.target_grasp_1 = self.cube_pos_1.copy()
        self.target_lift = self.cube_pos_1.copy()
        self.target_lift[2] += 0.1  #how high we lift
        self.target_height_2 = self.cube_pos_2.copy()
        self.target_height_2[2] += 0.1  #height we stack at
        
        self.pid_controller = PID(self.kp_precise, self.ki_precise, self.kd_precise, self.target_height_1)
        self.phase = 0  #0: move to red, 1: grab red, 2: lift red, 3: approach green, 4: release
        self.gripper_open = True
        self.grasp_start_time = 0
        self.grasp_duration = 0.5
        self.release_start_time = 0
        self.release_duration = 1
        
    def get_action(self, obs):
        eef_pos = obs['robot0_eef_pos']
        self.cube_pos_1 = obs['cubeA_pos']
        self.cube_pos_2 = obs['cubeB_pos']
        error = self.pid_controller.get_error()
        
        if (self.phase == 0 and error < 0.02):
            self.phase = 1
            self.pid_controller.reset(self.target_grasp_1)

        elif (self.phase == 1 and error < 0.01):
            self.phase = 2
            self.gripper_open = False
            self.grasp_start_time = time.time()

        elif (self.phase == 2 and self.grasp_duration < (time.time() - self.grasp_start_time)):
            self.phase = 3
            self.pid_controller = PID(self.kp_lift, self.ki_lift, self.kd_lift, self.target_lift)

        elif (self.phase == 3 and error < 0.03):
            self.phase = 4
            self.pid_controller = PID(self.kp_precise, self.ki_precise, self.kd_precise, self.target_height_2)

        elif (self.phase == 4 and error < 0.02):
            self.phase = 5
            self.pid_controller.reset(self.cube_pos_2 + np.array([0, 0, 0.075])) #set it down more gently
            self.release_start_time = time.time()

        elif (self.phase == 5):
            if (time.time() - self.release_start_time) > self.release_duration:
                self.phase = 6 #set to 6 to release gripper
                self.gripper_open = True
                self.settle_start_time = time.time()
        
        control = self.pid_controller.update(eef_pos, 0.01)
        
        action = np.zeros(7)
        action[:3] = control[:3]
        
        if 2 <= self.phase < 5: #keep gripper closed while moving
            action[-1] = 1
        else:
            action[-1] = -1 if self.gripper_open else 1
        
        return action

class DoorPolicy(object):
    def __init__(self, obs):
        """
        Robust door opening policy for opposite-side doors (right side of screen).
        Implements:
        1. Correct approach from the front-left side
        2. Firm grasp with downward pressure
        3. Proper downward-and-left turning motion
        """
        self.kp_approach = [8.0, 8.0, 8.0]
        self.ki_approach = [0.1, 0.1, 0.1]
        self.kd_approach = [0.5, 0.5, 0.5]
        self.kp_turn = [4.0, 4.0, 2.0]
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