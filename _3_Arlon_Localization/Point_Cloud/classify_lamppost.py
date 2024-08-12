# Read instances (.las) using laspy
# Create point cloud from LasDat to Open3D
import laspy
import os
import open3d as o3d
import numpy as np
from utils import *
lasDouble = laspy.read('references/Double.las')
ptsDouble = np.vstack([lasDouble.x, lasDouble.y, lasDouble.z])
ptsDouble=np.transpose(ptsDouble)
refDouble = o3d.utility.Vector3dVector(ptsDouble)

lasSingle = laspy.read('references/Single.las')
ptsSingle = np.transpose(np.vstack([lasSingle.x, lasSingle.y, lasSingle.z]))
refSingle = o3d.geometry.PointCloud()
refSingle = o3d.utility.Vector3dVector(ptsSingle)

lasDouble2 = laspy.read('references/Double2.las')
ptsDouble2 = np.transpose(np.vstack([lasDouble2.x, lasDouble2.y, lasDouble2.z]))
refDouble2 = o3d.geometry.PointCloud()
refDouble2 = o3d.utility.Vector3dVector(ptsDouble2)

lasSingle2 = laspy.read('references/Single2.las')
ptsSingle2 = np.transpose(np.vstack([lasSingle2.x, lasSingle2.y, lasSingle2.z]))
refSingle2 = o3d.geometry.PointCloud()
refSingle2 = o3d.utility.Vector3dVector(ptsSingle2)

lasCurved = laspy.read('references/Curved.las')
ptsCurved = np.transpose(np.vstack([lasCurved.x, lasCurved.y, lasCurved.z]))
refCurved = o3d.geometry.PointCloud()
refCurved = o3d.utility.Vector3dVector(ptsCurved)

las = laspy.read("C:/PFE_Roadmap/point-cloud/offset/final_segment_Hc_correct.las")
segments_id = np.unique(las.segmentID)

for id in segments_id:
    lasInstance = las[las.segmentID == id]
    points = np.transpose(np.vstack([lasInstance.x, lasInstance.y, lasInstance.z]))
    cloud = o3d.geometry.PointCloud()
    cloud = o3d.utility.Vector3dVector(points)

    rmse = []
    fitness = []
    for i, ref in enumerate([refDouble, refSingle,refDouble2, refSingle2,refCurved  ]):

        # Fast Global Registration and ICP refinement
        voxel_size = 0.05  # means 5cm for the dataset
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)
        result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
        distance_threshold = voxel_size * 0.4
        result = o3d.registration.registration_icp(source, target, distance_threshold, result_ransac.transformation, o3d.registration.TransformationEstimationPointToPlane())
        new_rmse = result.rmse
        new_fitness = result.fitness

        rmse.append(new_rmse)
        fitness.append(new_fitness)

    min_rmse = min(rmse)
    max_fitness = max(fitness)

    idRef_rmse = rmse.index(min_rmse)
    idRef_fitness = fitness.index(max_fitness)

    if idRef_rmse == 0:
        print("Double Pole")
    elif idRef_rmse == 1:
        print("Single Pole")
    else:
        print("None pole")