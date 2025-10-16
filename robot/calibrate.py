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
# 提示没有aruco的看问题汇总
import transforms3d as tfs

tcp_host_ip = '192.168.56.10' # IP and port to robot arm as TCP client (UR5)
tcp_port = 30003
tool_orientation = [-np.pi, 0, np.pi]
# ---------------------------------------------

# Move robot to home pose
robot = UR_Robot(tcp_host_ip,tcp_port,is_use_robotiq85=False,is_use_camera=False)

# realsense初始化,配置摄像头与开启pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

#获取jaka的末端位姿，xyz；弧度制rxryrz
def get_jaka_gripper():
    tcp_pos = robot.parse_tcp_state_data(robot.get_state(),'cartesian_info')
    tcp_pos[3:6]=robot.rotvec2rpy(tcp_pos[3:6])
    return  tcp_pos
# 获取对齐的rgb和深度图
def get_aligned_images():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    # 获取intelrealsense参数
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    # 内参矩阵，转ndarray方便后续opencv直接使用
    intr_matrix = np.array([
        [intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]
    ])
    # 深度图-16位
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    # 深度图-8位
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
    pos = np.where(depth_image_8bit == 0)
    depth_image_8bit[pos] = 255
    # rgb图
    color_image = np.asanyarray(color_frame.get_data())
    # return: rgb图，深度图，相机内参，相机畸变系数(intr.coeffs)
    return color_image, depth_image, intr_matrix, np.array(intr.coeffs)

#获取标定板位姿
def get_realsense_mark(intr_matrix, intr_coeffs):
    # 获取dictionary, 4x4的码，指示位50个
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    # 创建detector parameters
    parameters = aruco.DetectorParameters()
    # 输入rgb图, aruco的dictionary, 相机内参, 相机的畸变参数
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected_img_points = detector.detectMarkers(rgb)
    # 估计出aruco码的位姿，0.045对应markerLength参数，单位是meter
    # rvec是旋转向量， tvec是平移向量
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
        # 按键盘r记录g-b，m-c位姿
        elif key == ord('r'):
            hands.append(get_jaka_gripper())
            cameras.append(get_realsense_mark(intr_matrix,intr_coeffs))
            print("record_ok")
        elif key ==ord('c'):
            R_Hgs, R_Hcs = [], []
            T_Hgs, T_Hcs = [], []
            for camera in cameras:
                #m-c的旋转矩阵和位移矩阵
                c = camera[3:6]
                # R_Hcs.append(tfs.quaternions.quat2mat((q[3], q[0], q[1], q[2]))) #四元素转旋转矩阵；相机读出x,y,z,w 使用该方法
                camera_mat,j = cv2.Rodrigues((c[0],c[1],c[2])) #旋转矢量到旋转矩阵
                R_Hcs.append(camera_mat)
                T_Hcs.append(np.array(camera[0:3])*1000) #统一单位   *1000
            for hand in hands:
                # g-b的旋转矩阵和位移矩阵
                g = hand[3:6]
                #R_Hgs.append(tfs.euler.euler2mat(math.radians(g[0])... 'sxyz'))#如果读出角度，转弧度再计算
                R_Hgs.append(tfs.euler.euler2mat(g[0], g[1], g[2], 'sxyz'))#欧拉角到旋转矩阵；
                T_Hgs.append(np.array(hand[0:3])*1000)
            print("R_Hcs:",R_Hcs)
            print("T_Hcs:",T_Hcs)
            print("R_Hgs:",R_Hgs)
            print("T_Hgs:",T_Hgs)
            #计算c-g
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_Hgs, T_Hgs, R_Hcs, T_Hcs,
                                                               method=cv2.CALIB_HAND_EYE_PARK)
            RT_c2g = tfs.affines.compose(np.squeeze(t_cam2gripper), R_cam2gripper, [1, 1, 1])
            print("RT_c2g：",RT_c2g)

            #根据计算 RT_c2g 推算出之前记录数据的机械臂末端相对基地移动矩阵
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
