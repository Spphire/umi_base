import numpy as np
from scipy.spatial.transform import Rotation as R

def quat2eulerZYX(quat, degree=False):
    """
    Convert quaternion to Euler angles with ZYX axis rotations.
    Parameters
    ----------
    quat : list
        Quaternion input in [w,x,y,z] order.
    degree : bool
        Return values in degrees, otherwise in radians.
    Returns
    ----------
    list
        Euler angles in [x,y,z] order, radian by default unless specified otherwise.
    """
    # Convert quaternion to Euler ZYX
    eulerZYX = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz', degrees=degree).tolist()
    return eulerZYX

def parse_pt_states(pt_states, parse_target):
    """
    Parse the value of a specified primitive state from the pt_states string list.
    Parameters
    ----------
    pt_states : list of str
        Primitive states string list returned from Robot::getPrimitiveStates().
    parse_target : str
        Name of the primitive state to parse for.
    Returns
    ----------
    str
        Value of the specified primitive state in string format. Empty string is 
        returned if parse_target does not exist.
    """
    for state in pt_states:
        words = state.split()
        if words[0] == parse_target:
            return words[-1]
    return ""

def create_homogeneous_matrix(position, euler_angles):
    """
    Create homogeneous transformation matrix from position and euler angles.
    Parameters
    ----------
    position : list of float
        Position vector in [x, y, z] order.
    euler_angles : list of float
        Euler angles in [x, y, z] order, in degrees.
    Returns
    ----------
    numpy.ndarray
        4x4 homogeneous transformation matrix.
    """
    matrix = np.eye(4)
    rotation_matrix = euler_angles_to_rotation_matrix(euler_angles)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = position
    return matrix

def euler_angles_to_rotation_matrix(euler_angles, degrees=True):
    """
    Convert euler angles to rotation matrix.
    Parameters
    ----------
    euler_angles : float list
        Euler angles in [x, y, z] order.
    degrees : bool
        Whether the input angles are in degrees.
    Returns
    ----------
    numpy.ndarray
        3x3 rotation matrix.
    """
    rot = R.from_euler('xyz', euler_angles, degrees=degrees)
    return rot.as_matrix()

def rotation_matrix_to_euler_angles(rotation_matrix, degrees=True):
    """
    Convert rotation matrix to euler angles.
    Parameters
    ----------
    rotation_matrix : numpy.ndarray
        3x3 rotation matrix.
    degrees : bool
        Whether to return angles in degrees.
    Returns
    ----------
    numpy.ndarray
        Euler angles in [x, y, z] order.
    """
    rot = R.from_matrix(rotation_matrix)
    return rot.as_euler('xyz', degrees=degrees)

def list2str(ls):
    return ' '.join(map(str, ls)) + ' '