import sys
sys.path.append('/home/pmh/Code/dab_deformable_resnet_5d_Rotate_Point_QK_nobias/')
sys.path.append('/home/pmh/anaconda3/envs/robot/lib/python3.8/site-packages/cv2/')
from UR_Robot import UR_Robot
import math
import time
import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
# from cv2.aruco import estimatePoseSingleMarkers
# Tip: There is no aruco to see the problem summary
import transforms3d as tfs

tcp_host_ip = '192.168.56.10' # IP and port to robot arm as TCP client (UR5)
tcp_port = 30003
tool_orientation = [-np.pi, 0, np.pi]
# ---------------------------------------------

# Move robot to home pose
robot = UR_Robot(tcp_host_ip,tcp_port,is_use_robotiq85=False,is_use_camera=False)

# Realsense initialization, camera configuration and pipeline opening
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

#Get the end position of Jaka, xyz; radian system rxryrz
def get_jaka_gripper():
    tcp_pos = robot.parse_tcp_state_data(robot.get_state(),'cartesian_info')
    tcp_pos[3:6]=robot.rotvec2rpy(tcp_pos[3:6])
    return  tcp_pos
# Get aligned RGB and depth maps
def get_aligned_images():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    # Get Intel Real Sense parameters
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    # Internal parameter matrix, converted to ndarray for subsequent direct use in opencv
    intr_matrix = np.array([
        [intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]
    ])
    # Depth Map - 16 bit
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    # Depth Map - 8 bit
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
    pos = np.where(depth_image_8bit == 0)
    depth_image_8bit[pos] = 255
    # RGB image
    color_image = np.asanyarray(color_frame.get_data())
    # return: RGB image, depth map, camera internal parameters, camera distortion coefficient (intr.coeffs)
    return color_image, depth_image, intr_matrix, np.array(intr.coeffs)

#Get the calibration plate pose
def get_realsense_mark(intr_matrix, intr_coeffs):
    # Get dictionary, 4x4 code, 50 indicator bits
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    # Create detector parameters
    parameters = aruco.DetectorParameters()
    # Input RGB image, Aruco dictionary, camera internal parameters, camera distortion parameters
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected_img_points = detector.detectMarkers(rgb)
    # Estimate the pose of the aruco code, 0.045 corresponds to the markerLength parameter, the unit is meter
    # rvec is the rotation vector, tvec is the translation vector
    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.1, intr_matrix, intr_coeffs)
    # for i in range(rvec.shape[0]):
    #     cv2.drawFrameAxes(rgb, intr_matrix, intr_coeffs, rvec[i, :, :], tvec[i, :, :], 0.03)
    #     aruco.drawDetectedMarkers(rgb,corners)
    return list(np.reshape(tvec,3))+list(np.reshape(rvec,3))
if __name__ == "__main__":
    hands =[]
    cameras = []
    while True:
        rgb, depth, intr_matrix, intr_coeffs = get_aligned_images()
        # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cv2.imshow('RGB image', rgb)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            pipeline.stop()
            break
        # Press r on the keyboard to record g-b, m-c posture
        elif key == ord('r'):
            hands.append(get_jaka_gripper())
            cameras.append(get_realsense_mark(intr_matrix,intr_coeffs))
            print("record_ok")
        elif key ==ord('c'):
            R_Hgs, R_Hcs = [], []
            T_Hgs, T_Hcs = [], []
            for camera in cameras:
                # The rotation matrix and displacement matrix of m-c
                c = camera[3:6]
                # R_Hcs.append(tfs.quaternions.quat2mat((q[3], q[0], q[1], q[2]))) #Four-element rotation matrix; camera reads x, y, z, w using this method
                camera_mat,j = cv2.Rodrigues((c[0],c[1],c[2])) # Rotation vector to rotation matrix
                R_Hcs.append(camera_mat)
                T_Hcs.append(np.array(camera[0:3])*1000) # Unified Unit   *1000
            for hand in hands:
                # g-b rotation matrix and displacement matrix
                g = hand[3:6]
                #R_Hgs.append(tfs.euler.euler2mat(math.radians(g[0])... 'sxyz'))# If you read the angle, convert it to radians and then calculate
                R_Hgs.append(tfs.euler.euler2mat(g[0], g[1], g[2], 'sxyz'))# Euler angles to rotation matrix；
                T_Hgs.append(np.array(hand[0:3])*1000)
            print("R_Hcs:",R_Hcs)
            print("T_Hcs:",T_Hcs)
            print("R_Hgs:",R_Hgs)
            print("T_Hgs:",T_Hgs)
            # Calculate c-g
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_Hgs, T_Hgs, R_Hcs, T_Hcs,
                                                               method=cv2.CALIB_HAND_EYE_PARK)
            RT_c2g = tfs.affines.compose(np.squeeze(t_cam2gripper), R_cam2gripper, [1, 1, 1])
            print("RT_c2g：",RT_c2g)

            # Based on the calculation of RT_c2g, the movement matrix of the end of the manipulator relative to the base is calculated.
            final_pose = []
            for i in range(len(R_Hgs)):
                RT_g2b = tfs.affines.compose(np.squeeze(T_Hgs[i]), R_Hgs[i], [1, 1, 1])
                temp = np.dot(RT_g2b, RT_c2g)
                RT_t2c = tfs.affines.compose(np.squeeze(T_Hcs[i]), R_Hcs[i], [1, 1, 1])
                temp = np.dot(temp, RT_t2c)
                tr = temp[0:3, 3:4].T[0]
                rot = tfs.euler.mat2euler(temp[0:3, 0:3])
                final_pose.append([tr[0], tr[1], tr[2], math.degrees(rot[0]), rot[1], rot[2]])
            final_pose = np.array(final_pose)
            print('final_pose\n', final_pose)
            break
    cv2.destroyAllWindows()
