from post_process_scripts.utils.WidthFromAngle import width_from_angle_v1, width_from_angle_v2
from scipy.optimize import minimize_scalar, minimize, differential_evolution
from post_process_scripts.utils.Curves import get_dh_curve, get_robotiq_curve

def open_closed_error_v1(params, angles, widths, ref):
    open_angle, closed_angle = params
    # 防止除以0或 open > close
    if closed_angle <= open_angle:
        return 1e6
    err = 0.0
    for a, w in zip(angles, widths):
        pred = width_from_angle_v1(a, ref, open_angle, closed_angle)
        if pred is None:
            continue
        err += (pred - w) ** 2
    return err

def get_optimal_v1(
    gripperAngle: list[float],
    gripperWidth: list[float],
    ref: list[float] = None,
):
    if ref is None:
        ref = get_robotiq_curve()

    # 定义 bounds
    bounds = [(-100, 100), (-100, 100)]  # open_angle, closed_angle

    # 使用 differential_evolution 做全局优化
    res = differential_evolution(
        lambda p: open_closed_error_v1(p, gripperAngle, gripperWidth, ref),
        bounds=bounds,
        strategy='best1bin',
        maxiter=1000,
        tol=1e-6
    )

    return res.x, res.fun

def closed_angle_error_v2(
    closed_angle: float,
    angles: list[float],
    widths: list[float],
    ref: list,
) -> float:
    err = 0.0
    for a, w in zip(angles, widths):
        pred = width_from_angle_v2(a, ref, closed_angle)
        if pred is None:
            continue
        err += (pred - w) ** 2
    return err

def get_optimal_v2(
    gripperAngle: list[float],
    gripperWidth: list[float],
    ref: list[float] = get_robotiq_curve(),
):
    result = minimize_scalar(
        lambda ca: closed_angle_error_v2(ca, gripperAngle, gripperWidth, ref),
        bounds=(-100, 100),
        method="bounded"
    )

    return result.x, result.fun