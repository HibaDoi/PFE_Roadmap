import numpy as np
import pandas as pd 
def calculate_euler_angles_from_rotation_matrix(R):
    # Yaw (kappa), rotation around z-axis
    kappa = np.arctan2(R[1, 0], R[0, 0])
    # Pitch (phi), rotation around y-axis
    # Avoid gimbal lock by using safe sqrt in the denominator
    phi = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    # Roll (omega), rotation around x-axis
    omega = np.arctan2(R[2, 1], R[2, 2])
    # Convert radians to degrees
    omega_deg = np.degrees(omega)
    phi_deg = np.degrees(phi)
    kappa_deg = np.degrees(kappa)
    return omega_deg, phi_deg, kappa_deg
def grads_to_radians(grads):
    return grads * np.pi / 200.0
def rotation_matrix_intrinsic(omega, phi, kappa):
    # Convert grads to radians
    omega_rad = grads_to_radians(omega)
    phi_rad = grads_to_radians(phi)
    kappa_rad = grads_to_radians(kappa)

    # Rotation matrix around x-axis
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(omega_rad), -np.sin(omega_rad)],
                    [0, np.sin(omega_rad), np.cos(omega_rad)]])

    # Rotation matrix around y-axis
    R_y = np.array([[np.cos(phi_rad), 0, np.sin(phi_rad)],
                    [0, 1, 0],
                    [-np.sin(phi_rad), 0, np.cos(phi_rad)]])

    # Rotation matrix around z-axis
    R_z = np.array([[np.cos(kappa_rad), -np.sin(kappa_rad), 0],
                    [np.sin(kappa_rad), np.cos(kappa_rad), 0],
                    [0, 0, 1]])

    # Combined rotation matrix in intrinsic order
    R = np.dot(R_x, np.dot(R_y, R_z))
    print(R)
    return R
df = pd.read_csv('Arlon_Localization/camera_info.csv', sep=';')
for index, row in df.iterrows():
    Omega1=row['omega']
    Phi1=row['phi']
    Kappa1=row['kappa']
    R=rotation_matrix_intrinsic(Omega1, Phi1, Kappa1)
    R[:, [1, 2]] = R[:, [2, 1]]
    omega_deg, phi_deg, kappa_deg = calculate_euler_angles_from_rotation_matrix(R)
    df.loc[index, 'roll'] = omega_deg
    df.loc[index, 'Pitch'] = phi_deg
    df.loc[index, 'Yaw'] = kappa_deg

print(df[['omega','roll','phi','Pitch','kappa','Yaw']])
df.to_csv('Roll_pitch_Yaw_Arlon.csv')