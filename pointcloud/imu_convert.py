import numpy as np
import matplotlib.pyplot as plt
import os



def llh_to_ecef(lat, lon, alt):
    a = 6378137.0  # WGS 84 semi-major axis
    f = 1 / 298.257223563  # flattening
    eSq = 2*f - f*f

    # Calculate ECEF coordinates
    N = a / np.sqrt(1 - (np.sin(np.radians(lat)) ** 2) * eSq)
    x = (N + alt) * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    y = (N + alt) * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    z = (N * (1-eSq) + alt) * np.sin(np.radians(lat))

    return x, y, z

def ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt):
    ref_x, ref_y, ref_z = llh_to_ecef(ref_lat, ref_lon, ref_alt)

    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)

    rotation_matrix = np.array([
        [-np.sin(ref_lon_rad), np.cos(ref_lon_rad), 0],
        [-np.sin(ref_lat_rad) * np.cos(ref_lon_rad), -np.sin(ref_lat_rad) * np.sin(ref_lon_rad), np.cos(ref_lat_rad)],
        [np.cos(ref_lat_rad) * np.cos(ref_lon_rad), np.cos(ref_lat_rad) * np.sin(ref_lon_rad), np.sin(ref_lat_rad)]
    ])

    enu = np.dot(rotation_matrix, np.array([dx, dy, dz]))

    return enu[0], enu[1], enu[2]

def euler_to_rotation_matrix(roll, pitch, yaw):
    print(roll,pitch,yaw)
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    rotation_matrix_roll = np.array([[1, 0, 0],
                                     [0, np.cos(roll_rad), -np.sin(roll_rad)],
                                     [0, np.sin(roll_rad), np.cos(roll_rad)]])

    rotation_matrix_pitch = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                                      [0, 1, 0],
                                      [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

    rotation_matrix_yaw = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                                    [0, 0, 1]])

    rotation_matrix = rotation_matrix_yaw @ rotation_matrix_pitch @ rotation_matrix_roll
    
    return rotation_matrix

def convert(items, imu_to_vel):
    ref = items[0]
    transformations = []
    translations = []
    rotation_ref = euler_to_rotation_matrix(ref[3], ref[4], ref[5])
    for item in items:
        ecef_x, ecef_y, ecef_z = llh_to_ecef(item[0], item[1], item[2])
        enu_x, enu_y, enu_z = ecef_to_enu(ecef_x, ecef_y, ecef_z, ref[0], ref[1], ref[2])
        translations.append([enu_x, enu_y, enu_z])

        rotation_matrix = euler_to_rotation_matrix(item[3], item[4], item[5])
        rotation = np.linalg.inv(rotation_ref) @ rotation_matrix

        transformation = np.vstack((np.hstack((rotation, np.array([enu_x, enu_y, enu_z]).reshape(3,1))),[0, 0, 0, 1]))
        transformation = transformation @ np.linalg.inv(imu_to_vel)
        transformations.append(transformation)
        
    return transformations

def get_imu():
    file_path = './data/kitti/2011_09_26_drive_0005_sync/oxts/data'
    files = os.listdir(file_path)
    files.sort()

    items = []
    for file in files:
        with open(file_path + os.sep + file, 'r') as f:
            line = f.readline()
            items.append([float(val) for val in line.split()[:6]])

    cal_path = './data/kitti/2011_09_26_drive_0005_sync/calibration/calib_imu_to_velo.txt'
    with open(cal_path, 'r') as f2:
        vals = f2.readlines()
        rotation = np.array([float(val) for val in vals[1].split()[1:]]).reshape((3,3))
        translation = np.array([float(val) for val in vals[2].split()[1:]]).reshape((3,1))

        imu_to_vel = np.vstack((np.hstack((rotation, translation)), [0, 0, 0, 1]))

    transformations = convert(items, imu_to_vel)

    # transformations, translations = convert(items, imu_to_vel)
    # translations = np.array(translations)

    # fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    # ax.plot(translations[:,0],translations[:,1], translations[:,2])
    # plt.show()

    return transformations

if __name__ == '__main__':
    file_path = './data/kitti/2011_09_26_drive_0005_sync/oxts/data'
    files = os.listdir(file_path)
    files.sort()

    items = []
    for file in files:
        with open(file_path + os.sep + file, 'r') as f:
            line = f.readline()
            items.append([float(val) for val in line.split()[:6]])

    cal_path = './data/kitti/2011_09_26_drive_0005_sync/calibration/calib_imu_to_velo.txt'
    with open(cal_path, 'r') as f2:
        vals = f2.readlines()
        rotation = np.array([float(val) for val in vals[1].split()[1:]]).reshape((3,3))
        translation = np.array([float(val) for val in vals[2].split()[1:]]).reshape((3,1))

        imu_to_vel = np.vstack((np.hstack((rotation, translation)), [0, 0, 0, 1]))

    translations = convert(items, imu_to_vel)

    # transformations, translations = convert(items, imu_to_vel)
    translations = np.array(translations)

    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    ax.plot(translations[:,0],translations[:,1], translations[:,2])
    plt.show()