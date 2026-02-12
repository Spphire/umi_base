def width_from_angle_v1(
        angle: float,
        ref: list,
        open_angle: float | None,
        closed_angle: float | None,
) -> float | None:
    # assert ref length >= 2
    if len(ref) < 2:
        print("Reference data must contain at least two points")
        return None

    if closed_angle - open_angle == 0.0:
        print("Open and closed angle are the same")
        return None

    # normalize angle to reference range
    angle_ratio_from_open = (angle - open_angle) / (closed_angle - open_angle)
    aligned_angle = ref[0][0] + angle_ratio_from_open * (ref[-1][0] - ref[0][0])

    # clamp
    clamped_angle = max(min(aligned_angle, ref[-1][0]), ref[0][0])

    # find segment index i
    i = 0
    while i < len(ref) - 2 and ref[i + 1][0] < clamped_angle:
        i += 1

    # linear interpolation
    angle_ratio_from_i = (
        (clamped_angle - ref[i][0]) /
        (ref[i + 1][0] - ref[i][0])
    )

    width_mm = (
        ref[i][1] +
        angle_ratio_from_i * (ref[i + 1][1] - ref[i][1])
    )

    return width_mm / 1000.0

def width_from_angle_v2(
    angle: float,
    ref: list,
    closed_angle: float | None,
) -> float | None:
    # ref length check
    if len(ref) <= 2:
        print("Reference data must contain at least two points")
        return None

    if closed_angle is None:
        return None

    # 对齐角度
    aligned_angle = angle + (ref[-1][0] - closed_angle)

    # clamp 到 ref 的角度范围
    clamped_angle = max(
        min(aligned_angle, ref[-1][0]),
        ref[0][0]
    )

    # 找到区间 i，使：
    # ref[i].angle <= clamped_angle <= ref[i+1].angle
    i = 0
    while i < len(ref) - 2 and ref[i + 1][0] < clamped_angle:
        i += 1

    # 线性插值
    angle_ratio_from_i = (
        (clamped_angle - ref[i][0]) /
        (ref[i + 1][0] - ref[i][0])
    )

    width_mm = ref[i][1] + angle_ratio_from_i * (
        ref[i + 1][1] - ref[i][1]
    )

    # Swift 里最后是 /1000.0
    return width_mm / 1000.0