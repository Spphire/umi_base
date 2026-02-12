import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
def rel(action_tcp_pose_mat, base_tcp_pose_mat):
    global_transform = np.eye(4)[None]
    global_transform[0, :3, :3] = R.from_euler("x", -90, degrees=True).as_matrix()

    return base_tcp_pose_mat @ action_tcp_pose_mat

def only_y_inference(action_tcp_pose_mat, base_tcp_pose_mat):
    global_transform = np.eye(4)[None]
    global_transform[0, :3, :3] = R.from_euler("x", 90, degrees=True).as_matrix() # @ R.from_euler("z", 180, degrees=True).as_matrix()
    base_tcp_pose_mat = global_transform @ base_tcp_pose_mat
    # action_tcp_pose_mat = global_transform @ action_tcp_pose_mat
    # hack: only consider the y rotation
    only_y_rotation = R.from_matrix(base_tcp_pose_mat[0, :3, :3]).as_euler(
        "xzy", degrees=False
    )
    only_y_rotation[:2] = 0
    only_y_rotation = R.from_euler(
        "xzy", only_y_rotation, degrees=False
    ).as_matrix()
    base_tcp_pose_mat[0, :3, :3] = only_y_rotation
    return np.linalg.inv(global_transform) @ base_tcp_pose_mat @ action_tcp_pose_mat

def only_y_inference2(action_tcp_pose_mat, base_tcp_pose_mat):
    result_transform = np.eye(4)[None]
    result_transform[0, :3, 3] = (base_tcp_pose_mat @ action_tcp_pose_mat)[0, :3, 3]

    global_transform = np.eye(4)[None]
    global_transform[0, :3, :3] = R.from_euler("x", -90, degrees=True).as_matrix()
    base_tcp_pose_mat = global_transform @ base_tcp_pose_mat
    # action_tcp_pose_mat = global_transform @ action_tcp_pose_mat
    # hack: only consider the y rotation
    only_y_rotation = R.from_matrix(base_tcp_pose_mat[0, :3, :3]).as_euler(
        "xzy", degrees=False
    )
    only_y_rotation[:2] = 0
    only_y_rotation = R.from_euler(
        "xzy", only_y_rotation, degrees=False
    ).as_matrix()
    #base_tcp_pose_mat[0, :3, :3] = only_y_rotation
    result_transform[0, :3, :3] = (np.linalg.inv(global_transform) @ base_tcp_pose_mat @ action_tcp_pose_mat)[0, :3, :3]

    return result_transform


def draw_axes(ax, origin, rot_mat, length=0.05, prefix=""):
    rot_mat = np.asarray(rot_mat)
    if rot_mat.shape == (4, 4):
        rot_mat = rot_mat[:3, :3]
    if rot_mat.shape == (1, 4, 4):
        rot_mat = rot_mat[0, :3, :3]
    if rot_mat.shape != (3, 3):
        raise ValueError(f"rot_mat must be 3x3 or 4x4, got {rot_mat.shape}")

    axes = rot_mat @ np.eye(3)
    axes = axes / (np.linalg.norm(axes, axis=0, keepdims=True) + 1e-8)

    colors = ["r", "g", "b"]
    labels = ["X", "Y", "Z"]
    for i in range(3):
        vec = axes[:, i] * length
        ax.quiver(
            origin[0], origin[1], origin[2],
            vec[0], vec[1], vec[2],
            color=colors[i], arrow_length_ratio=0.2, pivot="tail"
        )
        ax.text(origin[0] + vec[0], origin[1] + vec[1], origin[2] + vec[2], f"{prefix}{labels[i]}")
    if prefix:
        ax.text(origin[0], origin[1], origin[2], prefix)


def main():
    # action_tcp_pose_mat: +y translation 0.05m
    action_tcp_pose_mat = np.eye(4)[None]
    action_tcp_pose_mat[0, :3, 3] = np.array([0.0, 0.05, 0.0])
    action_tcp_pose_mat[0, :3, :3] = R.from_euler("x", -30, degrees=True).as_matrix()

    # base_tcp_pose_mat: rotate 180 deg around z, translate -y 0.5m
    base_tcp_pose_mat = np.eye(4)[None]
    base_tcp_pose_mat[0, :3, :3] = R.from_euler("x", -30, degrees=True).as_matrix() @ R.from_euler("z", 180, degrees=True).as_matrix()
    base_tcp_pose_mat[0, :3, 3] = np.array([0.0, -0.5, 0.0])

    # Keep a copy for visualization (initial base)
    base_tcp_pose_mat_init = base_tcp_pose_mat.copy()

    # Setup interactive plot
    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Control parameters
    trans_step = 0.01  # translation step in meters
    rot_step = 5       # rotation step in degrees
    
    def update_plot():
        ax.clear()
        
        final_pose_mat = only_y_inference2(action_tcp_pose_mat, base_tcp_pose_mat_init)
        
        # Points to visualize
        origin = np.array([0.0, 0.0, 0.0])
        base_init_pos = base_tcp_pose_mat_init[0, :3, 3]
        final_pos = final_pose_mat[0, :3, 3]
        base_init_rot = base_tcp_pose_mat_init[0, :3, :3]
        final_rot = final_pose_mat[0, :3, :3]
        
        ax.scatter(*origin, color="k", label="Origin")
        ax.scatter(*base_init_pos, color="b", label="Base init", s=100)
        ax.scatter(*final_pos, color="r", label="Final action", s=100)
        
        # Draw local coordinate axes
        draw_axes(ax, origin, np.eye(3), length=0.05, prefix="O")
        draw_axes(ax, base_init_pos, base_init_rot, length=0.05, prefix="B")
        draw_axes(ax, final_pos, final_rot, length=0.05, prefix="F")
        
        # Connect points for clarity
        ax.plot([origin[0], base_init_pos[0]], [origin[1], base_init_pos[1]], [origin[2], base_init_pos[2]], "b--")
        ax.plot([base_init_pos[0], final_pos[0]], [base_init_pos[1], final_pos[1]], [base_init_pos[2], final_pos[2]], "r--")
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("only-y-inference test\n[WASD]=XY [ZX]=Z [IJKL]=Rot [UO]=RotZ [R]=Reset")
        ax.legend()
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.6, 0.1)
        ax.set_zlim(-0.2, 0.2)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    def on_key(event):
        nonlocal base_tcp_pose_mat_init
        
        # Translation controls (WASD + ZX)
        if event.key == 'w':  # +Y
            base_tcp_pose_mat_init[0, 1, 3] += trans_step
        elif event.key == 's':  # -Y
            base_tcp_pose_mat_init[0, 1, 3] -= trans_step
        elif event.key == 'a':  # -X
            base_tcp_pose_mat_init[0, 0, 3] -= trans_step
        elif event.key == 'd':  # +X
            base_tcp_pose_mat_init[0, 0, 3] += trans_step
        elif event.key == 'z':  # +Z
            base_tcp_pose_mat_init[0, 2, 3] += trans_step
        elif event.key == 'x':  # -Z
            base_tcp_pose_mat_init[0, 2, 3] -= trans_step
        
        # Rotation controls (IJKL + UO)
        elif event.key == 'i':  # +X rotation (pitch up)
            rot = R.from_euler('x', rot_step, degrees=True).as_matrix()
            base_tcp_pose_mat_init[0, :3, :3] = base_tcp_pose_mat_init[0, :3, :3] @ rot
        elif event.key == 'k':  # -X rotation (pitch down)
            rot = R.from_euler('x', -rot_step, degrees=True).as_matrix()
            base_tcp_pose_mat_init[0, :3, :3] = base_tcp_pose_mat_init[0, :3, :3] @ rot
        elif event.key == 'j':  # +Y rotation (yaw left)
            rot = R.from_euler('y', rot_step, degrees=True).as_matrix()
            base_tcp_pose_mat_init[0, :3, :3] = base_tcp_pose_mat_init[0, :3, :3] @ rot
        elif event.key == 'l':  # -Y rotation (yaw right)
            rot = R.from_euler('y', -rot_step, degrees=True).as_matrix()
            base_tcp_pose_mat_init[0, :3, :3] = base_tcp_pose_mat_init[0, :3, :3] @ rot
        elif event.key == 'u':  # +Z rotation (roll left)
            rot = R.from_euler('z', rot_step, degrees=True).as_matrix()
            base_tcp_pose_mat_init[0, :3, :3] = base_tcp_pose_mat_init[0, :3, :3] @ rot
        elif event.key == 'o':  # -Z rotation (roll right)
            rot = R.from_euler('z', -rot_step, degrees=True).as_matrix()
            base_tcp_pose_mat_init[0, :3, :3] = base_tcp_pose_mat_init[0, :3, :3] @ rot
        
        # Reset
        elif event.key == 'r':
            base_tcp_pose_mat_init = base_tcp_pose_mat.copy()
        
        else:
            return  # No update needed
        
        update_plot()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    update_plot()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
