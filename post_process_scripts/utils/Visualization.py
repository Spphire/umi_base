import numpy as np

from post_process_scripts.utils.BsonReader import load_bson_file
from post_process_scripts.utils.Curves import get_dh_curve, get_robotiq_curve
from post_process_scripts.utils.WidthFromAngle import width_from_angle_v1, width_from_angle_v2
from matplotlib import pyplot as plt

def visualize_v1(gripperAngle, gripperWidth, open_angle=0, closed_angle=25, curve = get_robotiq_curve()):
    plt.figure()
    plt.plot(gripperAngle, label='Gripper Angle')
    plt.plot(np.array(gripperWidth)*1000, label='Gripper Width')
    width_v1 = np.array([width_from_angle_v1(ga, curve, open_angle, closed_angle) for ga in gripperAngle])*1000
    print(f"min {min(width_v1)} {min(np.array(gripperWidth)*1000)}")
    plt.plot(width_v1, label='width_v1')
    plt.legend()
    plt.show()

def visualize_v2(gripperAngle, gripperWidth, closed_angle=25, curve = get_robotiq_curve()):
    plt.figure()
    plt.plot(gripperAngle, label='Gripper Angle')
    plt.plot(np.array(gripperWidth)*1000, label='Gripper Width')
    width_v2 = np.array([width_from_angle_v2(ga, curve, closed_angle) for ga in gripperAngle])*1000
    plt.plot(width_v2, label='width_v2')
    plt.legend()
    plt.show()

def visualize_bson(bson_path, closed_angle=25, open_angle=None):
    bson_data = load_bson_file(bson_path)
    plt.figure()
    plt.plot(bson_data['gripperAngle'], label='Gripper Angle')
    plt.plot(np.array(bson_data['gripperWidth']) * 1000, label='Gripper Width')
    if open_angle is None:
        width = np.array([width_from_angle_v2(ga, get_robotiq_curve(), closed_angle) for ga in bson_data['gripperAngle']]) * 1000
    else:
        width = np.array(
            [width_from_angle_v1(ga, get_robotiq_curve(),open_angle , closed_angle) for ga in bson_data['gripperAngle']]) * 1000
    plt.plot(width, label='width')
    plt.legend()
    plt.show()
