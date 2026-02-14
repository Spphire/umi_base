import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import transforms3d as t3d

iphone_camera_center = np.array([
    [1., 0., 0., -0.0635],
    [0, 1., 0., 0.024],
    [0, 0., 1., 0.005],
    [0, 0., 0., 1],
])

# ============================================================
# Intrinsics
# ============================================================

def build_camera_matrix_from_list(intrin):
    """
    intrin: [fx, fy, cx, cy]
    """
    fx, fy, cx, cy = intrin
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

# ============================================================
# Unity (LH) -> OpenCV (RH)
# ============================================================

def unity2zup_right_frame_batch(pos_quat):
    assert pos_quat.ndim == 2, "Expected shape (n, 7) for pos_quat"
    assert pos_quat.shape[1] == 7, "Expected shape (n, 7) for pos_quat"
    return np.array([unity2zup_right_frame(pq) for pq in pos_quat])

def unity2zup_right_frame(pos_quat):
    pos_quat = np.array(pos_quat) * np.array([1, -1, 1, 1, -1, 1, -1])
    rot_mat = t3d.quaternions.quat2mat(pos_quat[3:])
    pos_vec = pos_quat[:3]
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = pos_vec
    fit_mat = t3d.euler.axangle2mat([0, 1, 0], np.pi / 2)
    fit_mat = fit_mat @ t3d.euler.axangle2mat([0, 0, 1], -np.pi / 2)
    _ = np.eye(4)
    _[:3, :3] = fit_mat
    return _ @ T

def unity_pose_to_cv_T(pose):
    """
    pose: [x y z qw qx qy qz] in Unity left-handed
    return: 4x4 T_wc in OpenCV right-handed
    """
    x, y, z, qw, qx, qy, qz = pose

    t_u = np.array([x, y, z])
    r_u = R.from_quat([qx, qy, qz, qw])
    R_u = r_u.as_matrix()
    T_u = np.eye(4)
    T_u[:3, :3] = R_u
    T_u[:3, 3] = t_u

    # Unity -> CV scale flip (Y axis)
    S = np.diag([1, 1, -1, 1])

    T_cv = S @ T_u
    return T_cv

# ============================================================
# PnP
# ============================================================

def pnp_to_T(rvec, tvec):
    R_cm, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R_cm
    T[:3, 3] = tvec.flatten()
    return T


def detect_aruco_pnp_T(
    frame,
    K,
    dist,
    marker_length=0.045,
    target_ids=[0, 1],
    reproj_err_thresh=0.5
):
    """
    return:
        success: bool
        result: dict {marker_id: T_cm}
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )
    detector = cv2.aruco.ArucoDetector(
        aruco_dict, cv2.aruco.DetectorParameters()
    )

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return False, {}

    ids = ids.flatten()

    # ---------- ArUco 物理点 ----------
    half = marker_length / 2
    obj_pts = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float32)

    results = {}

    for i, mid in enumerate(ids):
        if mid not in target_ids:
            continue

        # ---------- 像素点 ----------
        img_pts = corners[i].reshape(-1, 2).astype(np.float32)

        # ---------- 去畸变 ----------
        img_pts_undist = cv2.undistortPoints(
            img_pts.reshape(-1, 1, 2),
            K,
            dist,
            P=K
        ).reshape(-1, 2)

        # ---------- 初始 PnP ----------
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts_undist,
            K,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not ok:
            continue

        # ---------- PnP refine ----------
        cv2.solvePnPRefineLM(
            obj_pts,
            img_pts_undist,
            K,
            None,
            rvec,
            tvec
        )

        # ---------- 重投影误差 ----------
        proj_pts, _ = cv2.projectPoints(
            obj_pts,
            rvec,
            tvec,
            K,
            None
        )
        proj_pts = proj_pts.reshape(-1, 2)

        reproj_err = np.linalg.norm(
            proj_pts - img_pts_undist,
            axis=1
        )
        mean_err = np.mean(reproj_err)

        if mean_err > reproj_err_thresh:
            continue

        # ---------- 保存结果 ----------
        results[int(mid)] = pnp_to_T(rvec, tvec)

    if len(results) == 0:
        return False, {}

    return True, results




# ============================================================
# Pose consistency (world)
# ============================================================

def world_pose_consistent(T1, T2,
                          trans_thresh=0.01,
                          angle_thresh_deg=2.0):
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]

    if np.linalg.norm(t1 - t2) > trans_thresh:
        return False

    r1 = R.from_matrix(T1[:3, :3])
    r2 = R.from_matrix(T2[:3, :3])

    angle = (r1.inv() * r2).magnitude() * 180 / np.pi
    return angle < angle_thresh_deg

# ============================================================
# Main pipeline
# ============================================================

def load_calib_npz(npz_path):
    data = np.load(npz_path)
    return (
        data["mtx_l"], data["dist_l"].reshape(-1),
        data["mtx_r"], data["dist_r"].reshape(-1)
    )

def run_aruco_world_pnp_verbose(data, ArUco_ids=[0,1]):
    result = {
        "OK": {
            "frame_idx": {
            },
            "ArUco_poses": {
            },  # ArUco 世界位姿
            "iphone_camera_poses": {
            },
        },
        "NO_ARUCO": [],
        "PNP_FAIL": [],
        "WORLD_MISMATCH": [],
        "cam_l_poses": [],  # 左眼相机世界位姿
        "cam_r_poses": [],  # 右眼相机世界位姿
    }
    for id in ArUco_ids:
        result["OK"]["frame_idx"][id] = []
        result["OK"]["ArUco_poses"][id] = []
    print(data.keys())
    N = min(len(data["leftCameraFrames"]), len(data["rightCameraFrames"]))

    K_l = build_camera_matrix_from_list(data["leftCameraIntrinsics"])
    K_r = build_camera_matrix_from_list(data["rightCameraIntrinsics"])
    dist_l = np.zeros(5)
    dist_r = np.zeros(5)

    for i in range(N):
        # 2️⃣ camera pose
        T_wc_l = unity2zup_right_frame(
            data["leftCameraPoses"][i]
        )

        T_wc_r = unity2zup_right_frame(
            data["rightCameraPoses"][i]
        )

        result["cam_l_poses"].append(T_wc_l)
        result["cam_r_poses"].append(T_wc_r)

        if i % 2 == 0:
            continue

        ok_l, aruco_result_l = detect_aruco_pnp_T(
            data["leftCameraFrames"][i],
            K_l,
            dist_l,
            target_ids=ArUco_ids,
        )
        ok_r, aruco_result_r = detect_aruco_pnp_T(
            data["rightCameraFrames"][i],
            K_r,
            dist_r,
            target_ids=ArUco_ids,
        )      

        # 1️⃣ 完全没检测到 ArUco
        if not ok_l or not ok_r:
            result["NO_ARUCO"].append(i)
            continue

        for ArUco_id in ArUco_ids:
            if ArUco_id not in aruco_result_l.keys() or ArUco_id not in aruco_result_r.keys():
                continue

            T_cm_l = aruco_result_l[ArUco_id]
            T_cm_r = aruco_result_r[ArUco_id]

            T_wm_l = T_wc_l @ T_cm_l
            T_wm_r = T_wc_r @ T_cm_r

            # 3️⃣ 左右眼不一致
            if not world_pose_consistent(T_wm_l, T_wm_r):
                result["WORLD_MISMATCH"].append(i)
                continue

            # 4️⃣ OK
            T = average_two_T(T_wm_l, T_wm_r)

            result["OK"]["frame_idx"][ArUco_id].append(i)
            result["OK"]["ArUco_poses"][ArUco_id].append(T)

    for ArUco_id in ArUco_ids:
        if len(result["OK"]["ArUco_poses"][ArUco_id])>0:
            result["OK"]["iphone_camera_poses"][ArUco_id] = np.array(result["OK"]["ArUco_poses"][ArUco_id]) @ iphone_camera_center
    result["cam_l_poses"] = np.asarray(result["cam_l_poses"])
    result["cam_r_poses"] = np.asarray(result["cam_r_poses"])

    return result

def draw_coordinate_frame(ax, T, length=0.05):
    """
    ax: matplotlib 3d axis
    T: 4x4 world transform
    length: axis length (meters)
    """
    origin = T[:3, 3]
    Rm = T[:3, :3]

    x_axis = origin + Rm[:, 0] * length
    y_axis = origin + Rm[:, 1] * length
    z_axis = origin + Rm[:, 2] * length

    ax.plot(
        [origin[0], x_axis[0]],
        [origin[1], x_axis[1]],
        [origin[2], x_axis[2]],
        color='r'
    )
    ax.plot(
        [origin[0], y_axis[0]],
        [origin[1], y_axis[1]],
        [origin[2], y_axis[2]],
        color='g'
    )
    ax.plot(
        [origin[0], z_axis[0]],
        [origin[1], z_axis[1]],
        [origin[2], z_axis[2]],
        color='b'
    )

def average_two_T(T1, T2):
    # --- 平移 ---
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]
    t_avg = 0.5 * (t1 + t2)

    # --- 旋转 ---
    r1 = R.from_matrix(T1[:3, :3])
    r2 = R.from_matrix(T2[:3, :3])

    q1 = r1.as_quat()  # (x, y, z, w)
    q2 = r2.as_quat()

    # 四元数符号对齐（非常重要）
    if np.dot(q1, q2) < 0:
        q2 = -q2

    q_avg = q1 + q2
    q_avg /= np.linalg.norm(q_avg)

    R_avg = R.from_quat(q_avg).as_matrix()

    # --- 组合 ---
    T_avg = np.eye(4)
    T_avg[:3, :3] = R_avg
    T_avg[:3, 3] = t_avg

    return T_avg

def poses_to_T(poses):
    """
    poses: (n, 7) -> [x, y, z, qw, qx, qy, qz]
    return: (n, 4, 4)
    """
    poses = np.asarray(poses)
    n = poses.shape[0]

    x, y, z = poses[:, 0], poses[:, 1], poses[:, 2]
    qw, qx, qy, qz = poses[:, 3], poses[:, 4], poses[:, 5], poses[:, 6]

    # ---- 四元数归一化 ----
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm

    # ---- 旋转矩阵 ----
    R = np.zeros((n, 3, 3))

    R[:, 0, 0] = 1 - 2*(qy*qy + qz*qz)
    R[:, 0, 1] = 2*(qx*qy - qz*qw)
    R[:, 0, 2] = 2*(qx*qz + qy*qw)

    R[:, 1, 0] = 2*(qx*qy + qz*qw)
    R[:, 1, 1] = 1 - 2*(qx*qx + qz*qz)
    R[:, 1, 2] = 2*(qy*qz - qx*qw)

    R[:, 2, 0] = 2*(qx*qz - qy*qw)
    R[:, 2, 1] = 2*(qy*qz + qx*qw)
    R[:, 2, 2] = 1 - 2*(qx*qx + qy*qy)

    # ---- 齐次矩阵 ----
    T = np.zeros((n, 4, 4))
    T[:, :3, :3] = R
    T[:, :3, 3] = np.stack([x, y, z], axis=1)
    T[:, 3, 3] = 1.0

    return T

def T_to_pose(T: np.ndarray) -> np.ndarray:
    """
    T: (n,4,4) 齐次矩阵
    return: (n,7) -> [x, y, z, qw, qx, qy, qz]
    """
    T = np.asarray(T)
    n = T.shape[0]

    poses = np.zeros((n, 7), dtype=np.float64)

    # 平移
    poses[:, :3] = T[:, :3, 3]

    # 旋转
    rot_mats = T[:, :3, :3]  # shape (n,3,3)
    quats = R.from_matrix(rot_mats).as_quat()  # shape (n,4), xyzw
    # 转成 [qw, qx, qy, qz]
    poses[:, 3] = quats[:, 3]  # qw
    poses[:, 4] = quats[:, 0]  # qx
    poses[:, 5] = quats[:, 1]  # qy
    poses[:, 6] = quats[:, 2]  # qz

    return poses

def visualize_aruco(
    result,
    stride=10,
    axis_len=0.05
):
    """
    stride: 每隔多少帧画一个坐标系
    axis_len: 坐标轴长度 (m)
    """
    poses = np.array(result["OK"]["ArUco_poses"][0])

    iphone_camera_poses_T =  result["OK"]["iphone_camera_poses"][0]

    cam_l_poses = np.array(result["cam_l_poses"])[result["OK"]["frame_idx"][0]]
    xyz_cam_l = cam_l_poses[:, :3, 3]

    if len(poses) == 0:
        print("No valid poses to visualize.")
        return

    xyz = poses[:, :3, 3]

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 1️⃣ 轨迹
    ax.plot(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        '-o', markersize=3, alpha=0.6,
        label="ArUco position"
    )

    # 1️⃣ 轨迹
    ax.plot(
        iphone_camera_poses_T[:, 0, 3], iphone_camera_poses_T[:, 1, 3], iphone_camera_poses_T[:, 2, 3],
        '-o', markersize=3, alpha=0.6,
        label="center position"
    )

    # ⭐ 左眼相机轨迹
    ax.plot(
        xyz_cam_l[:, 0],
        xyz_cam_l[:, 1],
        xyz_cam_l[:, 2],
        '-^',
        markersize=3,
        alpha=0.6,
        label="Left camera position"
    )

    # 2️⃣ 按时间上色
    sc = ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        c=np.arange(len(xyz)),
        cmap='viridis',
        s=15
    )
    plt.colorbar(sc, ax=ax, label="Frame order")

    # 3️⃣ 画旋转（坐标系）
    for i in range(0, len(poses), stride):
        draw_coordinate_frame(ax, poses[i], length=axis_len)

    #
    for i in range(0, len(iphone_camera_poses_T), stride):
        draw_coordinate_frame(ax, iphone_camera_poses_T[i], length=axis_len * 0.4)

    # ⭐ 左眼相机坐标系
    for i in range(0, len(cam_l_poses), stride):
        draw_coordinate_frame(ax, cam_l_poses[i], length=axis_len * 0.8)

    # 4️⃣ 世界坐标轴
    world_len = np.max(np.linalg.norm(xyz, axis=1)) * 0.2
    ax.quiver(0, 0, 0, world_len, 0, 0, color='r', linewidth=2)
    ax.quiver(0, 0, 0, 0, world_len, 0, color='g', linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, world_len, color='b', linewidth=2)
    ax.text(world_len, 0, 0, 'X', color='r')
    ax.text(0, world_len, 0, 'Y', color='g')
    ax.text(0, 0, world_len, 'Z', color='b')

    # ===== 自动设置 xyz 轴范围 =====
    xyz_min = np.concatenate([xyz,xyz_cam_l], axis=0).min(axis=0)
    xyz_max = np.concatenate([xyz,xyz_cam_l], axis=0).max(axis=0)
    xyz_center = (xyz_min + xyz_max) / 2

    # 最大跨度（保证三轴等比例）
    span = np.max(xyz_max - xyz_min)

    # 扩大一点范围（20%）
    span *= 1.2
    half = span / 2

    ax.set_xlim(xyz_center[0] - half, xyz_center[0] + half)
    ax.set_ylim(xyz_center[1] - half, xyz_center[1] + half)
    ax.set_zlim(xyz_center[2] - half, xyz_center[2] + half)


    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("ArUco World Pose (Position + Orientation)")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()

    plt.tight_layout()
    plt.show()
