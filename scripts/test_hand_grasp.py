import os
import mujoco
import numpy as np
import imageio

def test_hand_grasp():
    os.makedirs("videos/snapshots", exist_ok=True)
    m = mujoco.MjModel.from_xml_path("assets/scene_self_arrest.xml")
    d = mujoco.MjData(m)

    slope_angle = np.deg2rad(35.0)
    x0 = 4.0
    z0 = x0 * np.tan(slope_angle)
    d.qpos[:3] = np.array([x0, 0.0, z0]) + np.array([-np.sin(slope_angle), 0.0, np.cos(slope_angle)]) * 0.10
    d.qpos[3:7] = [0.8870108, 0.0, 0.4617486, 0.0]

    q = np.zeros(m.nu)
    # Legs
    q[3] = 0.55   # left knee
    q[4] = -0.55  # left toe lift
    q[9] = 0.55   # right knee
    q[10] = -0.55 # right toe lift
    q[14] = -0.20 # waist arch

    # Left Arm: Reach along torso to grasp the lower shaft near the left hip
    q[15] = 0.36566022  # left shoulder pitch
    q[16] = -0.07271793 # left shoulder roll
    q[17] = 0.96422086  # left shoulder yaw
    q[18] = 1.42814642  # left elbow flex
    q[19] = -0.44681711 # left wrist roll
    q[20] = 0.68492481  # left wrist pitch
    q[21] = 0.01446427  # left wrist yaw

    # Left Hand Fingers: Closed fist wrapped around shaft
    q[22] = -0.6  # thumb 0
    q[23] = 0.7   # thumb 1
    q[24] = 1.5   # thumb 2
    q[25] = -1.4  # middle 0
    q[26] = -1.6  # middle 1
    q[27] = -1.4  # index 0
    q[28] = -1.6  # index 1

    # Right Arm: Tuck right hand firmly over axe head & press downward
    q[29] = 1.22507183  # right shoulder pitch
    q[30] = 0.18769837  # right shoulder roll
    q[31] = -0.04645577 # right shoulder yaw
    q[32] = 0.57926320  # right elbow flex
    q[33] = -0.50737203 # right wrist roll
    q[34] = -1.26848045 # right wrist pitch
    q[35] = -1.04006872 # right wrist yaw

    # Right Hand Fingers: Closed fist wrapped around axe head
    q[36] = 0.5   # thumb 0
    q[37] = -0.6  # thumb 1
    q[38] = -1.4  # thumb 2
    q[39] = 1.3   # index 0
    q[40] = 1.5   # index 1
    q[41] = 1.3   # middle 0
    q[42] = 1.5   # middle 1

    d.qpos[7:] = q
    d.ctrl[:] = q
    d.qvel[0:3] = np.array([-np.cos(slope_angle), 0.0, -np.sin(slope_angle)]) * 4.5
    mujoco.mj_forward(m, d)

    renderer = mujoco.Renderer(m, height=720, width=1280)
    
    # Opposing close-ups make finger/shaft intersections easy to audit.
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    cam.distance = 0.95
    cam.elevation = -25
    for azimuth, filename in [
        (55, "hand_grasp_both.png"),
        (125, "hand_grasp_reverse.png"),
    ]:
        cam.azimuth = azimuth
        renderer.update_scene(d, camera=cam)
        imageio.imwrite(f"videos/snapshots/{filename}", renderer.render())
    renderer.close()
    print("Rendered updated hand_grasp_both.png")

if __name__ == "__main__":
    test_hand_grasp()
