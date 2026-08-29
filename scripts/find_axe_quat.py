import numpy as np
import mujoco
import imageio
import os
import re

def test_quats_shaft_down():
    os.makedirs("videos/snapshots", exist_ok=True)
    
    # We want:
    # 1. Shaft pointing down across torso towards hip (Z_axe -> -torso)
    # 2. Pick pointing into the snow slope (X_axe -> into slope)
    # 3. Head right inside right hand palm
    quats = [
        ("shaft_down_pick_in_1", [0.0, 1.0, 0.0, 0.0], [0.04, -0.01, 0.0]),
        ("shaft_down_pick_in_2", [0.0, 0.7071, 0.7071, 0.0], [0.04, -0.01, 0.0]),
        ("shaft_down_pick_in_3", [0.7071, 0.0, 0.0, 0.7071], [0.04, -0.01, 0.0]),
        ("shaft_down_pick_in_4", [0.0, 0.0, 1.0, 0.0], [0.04, -0.01, 0.0]),
        ("shaft_down_pick_in_5", [0.5, 0.5, -0.5, -0.5], [0.04, -0.01, 0.0]),
    ]
    
    with open("assets/unitree_g1/g1_with_axe.xml", "r") as f:
        original_xml = f.read()

    for name, q, p in quats:
        xml_mod = re.sub(
            r'<body name="ice_axe" pos="[^"]*" quat="[^"]*">',
            f'<body name="ice_axe" pos="{p[0]} {p[1]} {p[2]}" quat="{q[0]} {q[1]} {q[2]} {q[3]}">',
            original_xml
        )
        with open("assets/unitree_g1/g1_with_axe.xml", "w") as f:
            f.write(xml_mod)
            
        m = mujoco.MjModel.from_xml_path("assets/scene_self_arrest.xml")
        d = mujoco.MjData(m)
        
        slope_angle = np.deg2rad(35.0)
        x0 = 4.0
        z0 = x0 * np.tan(slope_angle)
        d.qpos[:3] = np.array([x0, 0.0, z0]) + np.array([-np.sin(slope_angle), 0.0, np.cos(slope_angle)]) * 0.09
        d.qpos[3:7] = [0.88701, 0.0, 0.46175, 0.0]
        
        q_joints = m.key_qpos[0][7:].copy() if m.nkey > 0 else np.zeros(29)
        # Right arm:
        q_joints[22] = 0.6  # pitch
        q_joints[23] = -0.5 # roll in
        q_joints[24] = 0.4  # yaw
        q_joints[25] = 1.6  # elbow flex
        q_joints[26] = -0.5 # wrist roll
        q_joints[27] = -0.2 # wrist pitch
        q_joints[28] = 0.3  # wrist yaw
        
        # Left arm:
        q_joints[15] = 0.6
        q_joints[16] = 0.5
        q_joints[17] = -0.4
        q_joints[18] = 1.5
        
        # Legs:
        q_joints[3] = 0.4
        q_joints[9] = 0.4
        q_joints[4] = -0.4
        q_joints[10] = -0.4
        
        d.qpos[7:] = q_joints
        d.ctrl[:] = q_joints
        mujoco.mj_forward(m, d)
        
        renderer = mujoco.Renderer(m, height=720, width=1280)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        cam.distance = 2.0
        cam.azimuth = 80
        cam.elevation = -15
        
        renderer.update_scene(d, camera=cam)
        img = renderer.render()
        imageio.imwrite(f"videos/snapshots/test_{name}.png", img)
        renderer.close()
        print(f"Rendered test_{name}.png")

if __name__ == "__main__":
    test_quats_shaft_down()
