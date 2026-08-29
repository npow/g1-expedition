import math
import os

def create_ice_axe_obj(output_dir="assets/ice_axe"):
    os.makedirs(output_dir, exist_ok=True)
    obj_path = os.path.join(output_dir, "ice_axe.obj")
    mtl_path = os.path.join(output_dir, "ice_axe.mtl")

    # Generate MTL
    mtl_content = """# Ice Axe Materials
newmtl SteelHead
Ka 0.2 0.2 0.2
Kd 0.7 0.72 0.75
Ks 0.9 0.9 0.9
Ns 60.0
d 1.0

newmtl OrangeShaft
Ka 0.3 0.1 0.0
Kd 0.95 0.35 0.05
Ks 0.6 0.6 0.6
Ns 35.0
d 1.0

newmtl RubberGrip
Ka 0.05 0.05 0.05
Kd 0.15 0.15 0.15
Ks 0.1 0.1 0.1
Ns 5.0
d 1.0

newmtl SteelSpike
Ka 0.2 0.2 0.2
Kd 0.8 0.8 0.85
Ks 0.95 0.95 0.95
Ns 80.0
d 1.0
"""
    with open(mtl_path, "w") as f:
        f.write(mtl_content)

    vertices = []
    normals = []
    faces = [] # (material, [(v_idx), ...])

    def add_cylinder(p1, p2, r, n_segments=16, material="OrangeShaft"):
        dx, dy, dz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        vx, vy, vz = dx/length, dy/length, dz/length
        
        # orthogonal vectors
        not_v = (1.0, 0.0, 0.0) if abs(vx) < 0.8 else (0.0, 1.0, 0.0)
        # u = v x not_v
        ux = vy * not_v[2] - vz * not_v[1]
        uy = vz * not_v[0] - vx * not_v[2]
        uz = vx * not_v[1] - vy * not_v[0]
        u_len = math.sqrt(ux*ux + uy*uy + uz*uz)
        ux, uy, uz = ux/u_len, uy/u_len, uz/u_len
        # v_vec = v x u
        wx = vy * uz - vz * uy
        wy = vz * ux - vx * uz
        wz = vx * uy - vy * ux

        thetas = [2 * math.pi * i / n_segments for i in range(n_segments)]
        base_v_idx = len(vertices) + 1

        # Circle 1 (p1) and Circle 2 (p2)
        for t in thetas:
            ct, st = math.cos(t), math.sin(t)
            ox = r * (ct * ux + st * wx)
            oy = r * (ct * uy + st * wy)
            oz = r * (ct * uz + st * wz)
            vertices.append((p1[0] + ox, p1[1] + oy, p1[2] + oz))
            normals.append((ox/r, oy/r, oz/r))
        for t in thetas:
            ct, st = math.cos(t), math.sin(t)
            ox = r * (ct * ux + st * wx)
            oy = r * (ct * uy + st * wy)
            oz = r * (ct * uz + st * wz)
            vertices.append((p2[0] + ox, p2[1] + oy, p2[2] + oz))
            normals.append((ox/r, oy/r, oz/r))

        # Side faces
        for i in range(n_segments):
            next_i = (i + 1) % n_segments
            v1 = base_v_idx + i
            v2 = base_v_idx + next_i
            v3 = base_v_idx + n_segments + next_i
            v4 = base_v_idx + n_segments + i
            faces.append((material, [v1, v2, v3, v4]))

    def add_box(center, size, material="SteelHead"):
        cx, cy, cz = center
        dx, dy, dz = size[0]/2.0, size[1]/2.0, size[2]/2.0
        base = len(vertices) + 1
        
        corners = [
            (cx-dx, cy-dy, cz-dz), (cx+dx, cy-dy, cz-dz),
            (cx+dx, cy+dy, cz-dz), (cx-dx, cy+dy, cz-dz),
            (cx-dx, cy-dy, cz+dz), (cx+dx, cy-dy, cz+dz),
            (cx+dx, cy+dy, cz+dz), (cx-dx, cy+dy, cz+dz),
        ]
        for c in corners:
            vertices.append(c)
            normals.append((0.0, 1.0, 0.0))

        box_faces = [
            [base+0, base+1, base+2, base+3], # bottom
            [base+4, base+7, base+6, base+5], # top
            [base+0, base+4, base+5, base+1], # front
            [base+2, base+6, base+7, base+3], # back
            [base+0, base+3, base+7, base+4], # left
            [base+1, base+5, base+6, base+2], # right
        ]
        for bf in box_faces:
            faces.append((material, bf))

    # --- Construct Ice Axe Parts ---
    # Shaft
    add_cylinder((0, 0, 0.05), (0, 0, 0.58), r=0.014, n_segments=16, material="OrangeShaft")
    
    # Grip (lower 0.22m of shaft)
    add_cylinder((0, 0, 0.05), (0, 0, 0.25), r=0.016, n_segments=16, material="RubberGrip")

    # Spike at bottom (Z = 0.0 to 0.05)
    spike_base = len(vertices) + 1
    vertices.append((0.0, 0.0, 0.0)) # tip
    normals.append((0.0, 0.0, -1.0))
    n_seg = 12
    thetas = [2 * math.pi * i / n_seg for i in range(n_seg)]
    for t in thetas:
        vertices.append((0.012 * math.cos(t), 0.012 * math.sin(t), 0.05))
        normals.append((math.cos(t), math.sin(t), 0.0))
    for i in range(n_seg):
        next_i = (i + 1) % n_seg
        faces.append(("SteelSpike", [spike_base, spike_base + 1 + i, spike_base + 1 + next_i]))

    # Axe Head Center
    add_box((0, 0, 0.60), (0.032, 0.035, 0.045), material="SteelHead")

    # Pick (curved downwards)
    pick_pts = [
        (0.016, 0.005, 0.605),
        (0.045, 0.0045, 0.598),
        (0.075, 0.004,  0.585),
        (0.100, 0.0035, 0.565),
        (0.120, 0.003,  0.540),
        (0.135, 0.001,  0.515), # sharp pick tip
    ]
    pick_heights = [0.032, 0.028, 0.024, 0.018, 0.014, 0.004]

    base_pick = len(vertices) + 1
    for (x, yh, z), h in zip(pick_pts, pick_heights):
        vertices.append((x, -yh, z + h/2))
        vertices.append((x,  yh, z + h/2))
        vertices.append((x,  yh, z - h/2))
        vertices.append((x, -yh, z - h/2))
        for _ in range(4):
            normals.append((1.0, 0.0, 0.0))

    for s in range(len(pick_pts) - 1):
        s1 = base_pick + s * 4
        s2 = base_pick + (s + 1) * 4
        faces.append(("SteelHead", [s1+0, s2+0, s2+1, s1+1]))
        faces.append(("SteelHead", [s1+1, s2+1, s2+2, s1+2]))
        faces.append(("SteelHead", [s1+2, s2+2, s2+3, s1+3]))
        faces.append(("SteelHead", [s1+3, s2+3, s2+0, s1+0]))
    
    tip_idx = base_pick + (len(pick_pts)-1)*4
    faces.append(("SteelHead", [tip_idx+0, tip_idx+1, tip_idx+2, tip_idx+3]))

    # Adze (rear blade)
    adze_pts = [
        (-0.016, 0.025, 0.605),
        (-0.045, 0.032, 0.608),
        (-0.075, 0.042, 0.600),
        (-0.095, 0.050, 0.585),
    ]
    adze_thick = 0.005
    base_adze = len(vertices) + 1
    for (x, w, z) in adze_pts:
        vertices.append((x, -w/2, z + adze_thick/2))
        vertices.append((x,  w/2, z + adze_thick/2))
        vertices.append((x,  w/2, z - adze_thick/2))
        vertices.append((x, -w/2, z - adze_thick/2))
        for _ in range(4):
            normals.append((-1.0, 0.0, 0.0))

    for s in range(len(adze_pts) - 1):
        s1 = base_adze + s * 4
        s2 = base_adze + (s + 1) * 4
        faces.append(("SteelHead", [s1+0, s2+0, s2+1, s1+1]))
        faces.append(("SteelHead", [s1+1, s2+1, s2+2, s1+2]))
        faces.append(("SteelHead", [s1+2, s2+2, s2+3, s1+3]))
        faces.append(("SteelHead", [s1+3, s2+3, s2+0, s1+0]))
    tip_adze = base_adze + (len(adze_pts)-1)*4
    faces.append(("SteelHead", [tip_adze+0, tip_adze+1, tip_adze+2, tip_adze+3]))

    # Write OBJ
    with open(obj_path, "w") as f:
        f.write("# Mountaineering Ice Axe 3D Mesh\n")
        f.write("mtllib ice_axe.mtl\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")

        current_mat = None
        for mat, face in faces:
            if mat != current_mat:
                f.write(f"usemtl {mat}\n")
                current_mat = mat
            f_str = " ".join([f"{idx}" for idx in face])
            f.write(f"f {f_str}\n")

    print(f"Successfully generated realistic ice axe at: {obj_path} ({len(vertices)} vertices, {len(faces)} faces)")

if __name__ == "__main__":
    create_ice_axe_obj()
