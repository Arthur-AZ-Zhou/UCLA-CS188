import numpy as np
import robosuite as suite
from policies import *
import cv2

"""
CA2: fill in the following functions for camera calibration
"""

def detect_red_blob(input_image):
    """
    Detects the largest red-colored blob in an RGB image.

    Parameters:
        input_image (np.ndarray): Input RGB image (BGR format as used by OpenCV).

    Returns:
        tuple or None: (x, y) coordinates of the red blob's centroid in pixel space, or None if no blob is found.
    """
    pass


def pixel_to_camera3d(x, y, depth_image, cam_intrinsics):
    """
    Converts a 2D pixel coordinate (x, y) and its depth value into a 3D point in the camera coordinate frame.

    Parameters:
        x (int): Pixel x-coordinate.
        y (int): Pixel y-coordinate.
        depth_image (np.ndarray): Depth map where depth_image[y, x] is the depth at the given pixel.
        cam_intrinsics (dict): Camera intrinsic parameters with keys 'fx', 'fy', 'cx', 'cy'.

    Returns:
        np.ndarray or None: 3D point [X, Y, Z] in the camera frame, or None if depth is zero at the pixel.
    """
    pass


def solve_for_rigid_transformation(inpts, outpts):
    """
    Solves for the rigid transformation (rotation and translation) that best aligns two sets of 3D points.

    Parameters:
        inpts (np.ndarray): Nx3 array of source points (e.g., in camera frame).
        outpts (np.ndarray): Nx3 array of destination points (e.g., in robot frame).

    Returns:
        np.ndarray: A 3x4 transformation matrix T such that outpt ≈ T * inpt (homogeneous form).
    """
    pass




def get_calibration_offset():
    """
    Returns a fixed manually-tuned 3D offset vector to apply after camera-to-robot calibration.

    This offset can be used to fine-tune the robot's end-effector position relative to the detected
    3D target (e.g., to account for gripper geometry).

    Returns:
        np.ndarray: A 1D array of shape (3,) representing [x, y, z] offset in meters.
    """
    return -np.array([0.05, 0.0, 0.015]) # change this number based on your calibration



def run_hand_eye_calibration():
    """
    Runs a calibration procedure using a simulated robot arm to detect a red blob in the scene,
    map its camera pixel coordinates to 3D space, and compute the rigid transformation matrix
    between the camera and robot coordinate frames.
    """
    # -------------------------
    # Environment Setup
    # -------------------------
    env = suite.make(
        env_name="Lift", 
        robots="Panda",  
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview", 
        camera_depths=True,
        ignore_done=True
    )

    # -------------------------
    # Camera Intrinsics
    # -------------------------
    fovy = env.sim.model.cam_fovy[env.sim.model.camera_name2id("agentview")]
    img_height = 256
    img_width = 256
    fy = 0.5 * img_height / np.tan(0.5 * fovy * np.pi / 180)
    fx = fy
    cx = img_width / 2
    cy = img_height / 2
    cam_intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    print(cam_intrinsics)

    # -------------------------
    # Run Lift Policy to pick up red block
    # -------------------------
    obs = env.reset()
    policy = LiftPolicy(obs['cube_pos'])

    while True:
        action = policy.get_action_lowdim(obs)
        obs, reward, _, _ = env.step(action)
        obs_rgb = cv2.cvtColor(obs['agentview_image'], cv2.COLOR_BGR2RGB)
        obs_rgb = cv2.flip(obs_rgb, 0)
        obs_rgb = cv2.flip(obs_rgb, 1)

        cv2.imshow('obs', obs_rgb)
        cv2.waitKey(1)

        if reward == 1.0:
            break

    # -------------------------
    # Sample Waypoints & Detect Red Blob
    # -------------------------
    waypoints = []
    for x in np.linspace(-0.10, 0.06, num=4):
        for y in np.linspace(-0.12, 0.12, num=3):
            for z in np.linspace(obs['cube_pos'][2] + 0.08, obs['cube_pos'][2] + 0.12, num=2):
                waypoints.append((x, y, z))

    camera_waypoints = []
    robot_waypoits = []

    for waypoint in waypoints:
        movto_policy = MoveToPolicy(waypoint)
        for _ in range(80):
            action = movto_policy.get_action(obs["robot0_eef_pos"])
            obs, reward, _, _ = env.step(action)

            obs_rgb = cv2.cvtColor(obs['agentview_image'], cv2.COLOR_BGR2RGB)
            obs_depth = obs['agentview_depth']
            obs_rgb = cv2.flip(obs_rgb, 0)
            obs_rgb = cv2.flip(obs_rgb, 1)
            obs_depth = cv2.flip(obs_depth, 0)
            obs_depth = cv2.flip(obs_depth, 1)

            cv2.imshow('obs', obs_rgb)
            cv2.waitKey(1)

        red_target = detect_red_blob(obs_rgb)
        if red_target:
            cv2.circle(obs_rgb, red_target, 5, (0, 255, 0), -1)
            cx, cy = red_target
            target_3d = pixel_to_camera3d(cx, cy, obs_depth, cam_intrinsics)
            print(target_3d)

            if target_3d is not None:
                camera_waypoints.append(target_3d)
                robot_waypoits.append(obs["robot0_eef_pos"])

            cv2.imshow('obs', obs_rgb)
            cv2.waitKey(1)

        print(waypoint, obs["robot0_eef_pos"], red_target)

    # -------------------------
    # Solve & Save Transformation
    # -------------------------
    camera_waypoints = np.array(camera_waypoints)
    robot_waypoits = np.array(robot_waypoits)

    T_CR = solve_for_rigid_transformation(camera_waypoints, robot_waypoits)
    print(T_CR)
    np.save("T_CR.npy", T_CR)
    return T_CR


if __name__ == "__main__":
    run_hand_eye_calibration()
