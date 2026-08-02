# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2017
# Arno Knapitsch <arno.knapitsch@gmail.com >
# Jaesik Park <syncle@gmail.com>
# Qian-Yi Zhou <Qianyi.Zhou@gmail.com>
# Vladlen Koltun <vkoltun@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# This python script is for downloading dataset from www.tanksandtemples.org
# The dataset has a different license, please refer to
# https://tanksandtemples.org/license/

# this script requires Open3D python binding
# please follow the intructions in setup.py before running this script.
import numpy as np
import open3d as o3d
import os
import argparse
import copy
# import torch

from config import scenes_tau_dict
from registration import (
    trajectory_alignment,
    registration_vol_ds,
    registration_unif,
    read_trajectory,
)
# from help_func import auto_orient_and_center_poses
from trajectory_io import CameraPose
from evaluation import EvaluateHisto
from util import make_dir
from plot import plot_graph


def run_evaluation(dataset_dir, traj_path, ply_path, out_dir, view_crop, num_points=-1):
    scene = os.path.basename(os.path.normpath(dataset_dir))

    if scene not in scenes_tau_dict:
        print(dataset_dir, scene)
        raise Exception("invalid dataset-dir, not in scenes_tau_dict")

    print("")
    print("===========================")
    print("Evaluating %s" % scene)
    print("===========================")

    dTau = scenes_tau_dict[scene]
    # put the crop-file, the GT file, the COLMAP SfM log file and
    # the alignment of the according scene in a folder of
    # the same scene name in the dataset_dir
    colmap_ref_logfile = os.path.join(dataset_dir, scene + "_COLMAP_SfM.log")

    # this is for groundtruth pointcloud, we can use it
    alignment = os.path.join(dataset_dir, scene + "_trans.txt")
    gt_filen = os.path.join(dataset_dir, scene + ".ply")
    # this crop file is also w.r.t the groundtruth pointcloud, we can use it. 
    # Otherwise we have to crop the estimated pointcloud by ourself
    cropfile = os.path.join(dataset_dir, scene + ".json")
    # this is not so necessary
    map_file = os.path.join(dataset_dir, scene + "_mapping_reference.txt")
    if not os.path.isfile(map_file):
        map_file = None
    map_file = None

    make_dir(out_dir)

    # Load reconstruction
    print(ply_path)
    o3d_mesh = o3d.io.read_triangle_mesh(ply_path)
    has_mesh = o3d_mesh.has_triangles()
    
    # We ALWAYS need a lightweight point cloud for the ICP registration iterations
    # to keep the alignment fast before we do the heavy sampling.
    pcd = o3d.io.read_point_cloud(ply_path)
    if has_mesh:
        import trimesh
        mesh = trimesh.load_mesh(ply_path)
        sampled_vertices = mesh.vertices[mesh.faces].mean(axis=1)
        vertices = np.concatenate([mesh.vertices, sampled_vertices], axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
    
    print(gt_filen)
    gt_pcd = o3d.io.read_point_cloud(gt_filen)

    gt_trans = np.loadtxt(alignment)
    print(traj_path)
    traj_to_register = []
    if traj_path.endswith('.npy'):
        ld = np.load(traj_path)
        for i in range(len(ld)):
            traj_to_register.append(CameraPose(meta=None, mat=ld[i]))
    elif traj_path.endswith('.json'): # instant-npg or sdfstudio format
        import json
        with open(traj_path, encoding='UTF-8') as f:
            meta = json.load(f)
        poses_dict = {}
        for i, frame in enumerate(meta['frames']):
            filepath = frame['file_path']
            new_i = int(filepath[13:18]) - 1
            poses_dict[new_i] = np.array(frame['transform_matrix'])
        poses = []
        for i in range(len(poses_dict)):
            poses.append(poses_dict[i])
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, _ = auto_orient_and_center_poses(poses, method='up', center_poses=True)
        scale_factor = 1.0 / float(torch.max(torch.abs(poses[:, :3, 3])))
        poses[:, :3, 3] *= scale_factor
        poses = poses.numpy()
        for i in range(len(poses)):
            traj_to_register.append(CameraPose(meta=None, mat=poses[i]))

    else:
        traj_to_register = read_trajectory(traj_path)
    print(colmap_ref_logfile)
    gt_traj_col = read_trajectory(colmap_ref_logfile)

    trajectory_transform = trajectory_alignment(map_file, traj_to_register,
                                                gt_traj_col, gt_trans, scene)

    
    # big pointclouds will be downlsampled to this number to speed up alignment
    dist_threshold = dTau
    # Refine alignment by using the actual GT and MVS pointclouds
    vol = o3d.visualization.read_selection_polygon_volume(cropfile)
    
    # Registration refinment in 3 iterations
    r2 = registration_vol_ds(pcd, gt_pcd, trajectory_transform, vol, dTau,
                             dTau * 80, 20)
    r3 = registration_vol_ds(pcd, gt_pcd, r2.transformation, vol, dTau / 2.0,
                             dTau * 20, 20)
    r = registration_unif(pcd, gt_pcd, r3.transformation, vol, 2 * dTau, 20)
    trajectory_transform = r.transformation
    
    
    # ==============================================================================
    # NEW LOGIC: Transform -> Crop Mesh -> Sample Points -> Evaluate
    # ==============================================================================
    if has_mesh and num_points > 0:
        print("Transforming and cropping the mesh BEFORE dense sampling...")
        o3d_mesh_copy = copy.deepcopy(o3d_mesh)
        o3d_mesh_copy.transform(trajectory_transform)
        
        try:
            cropped_mesh = vol.crop_triangle_mesh(o3d_mesh_copy)
            crop_success = True
        except Exception as e:
            print(f"Warning: Mesh cropping failed ({e}). Falling back to sampling before cropping.")
            crop_success = False

        if crop_success:
            if not cropped_mesh.has_triangles():
                print("Warning: Cropped mesh has no triangles left! Falling back to uncropped point cloud.")
                eval_pcd = copy.deepcopy(pcd)
                eval_pcd.transform(trajectory_transform)
            else:
                print(f"Sampling {num_points} points uniformly from the CROPPED mesh surface...")
                eval_pcd = cropped_mesh.sample_points_uniformly(number_of_points=num_points)
            
            # eval_pcd is already transformed to GT space, pass identity matrix to EvaluateHisto
            eval_trans = np.identity(4)
        else:
            # Fallback if crop_triangle_mesh isn't supported by Open3D version
            print(f"Sampling {num_points} points uniformly from the ENTIRE mesh surface...")
            eval_pcd = o3d_mesh.sample_points_uniformly(number_of_points=num_points)
            eval_trans = trajectory_transform
    else:
        eval_pcd = pcd
        eval_trans = trajectory_transform
    # ==============================================================================


    # Histogramms and P/R/F1
    plot_stretch = 5
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
        mean_recon2gt,
        mean_gt2recon,
        chamfer_dist
    ] = EvaluateHisto(
        eval_pcd,
        gt_pcd,
        eval_trans,
        vol,
        dTau / 2.0,
        dTau,
        out_dir,
        plot_stretch,
        scene,
        view_crop
    )
    eva = [precision, recall, fscore]
    print("==============================")
    print("evaluation result : %s" % scene)
    print("==============================")
    print("distance tau : %.3f" % dTau)
    print("precision : %.4f" % eva[0])
    print("recall : %.4f" % eva[1])
    print("f-score : %.4f" % eva[2])
    print("reconstruction to gt : %.6f" % mean_recon2gt)
    print("gt to reconstruction : %.6f" % mean_gt2recon)
    print("chamfer distance : %.6f" % chamfer_dist)
    print("==============================")
    
    # Save evaluation results to file
    with open(os.path.join(out_dir, "evaluation_results.txt"), "w") as f:
        f.write("==============================\n")
        f.write("evaluation result : %s\n" % scene)
        f.write("==============================\n")
        f.write("distance tau : %.3f\n" % dTau)
        f.write("precision : %.4f\n" % eva[0])
        f.write("recall : %.4f\n" % eva[1])
        f.write("f-score : %.4f\n" % eva[2])
        f.write("reconstruction to gt : %.6f\n" % mean_recon2gt)
        f.write("gt to reconstruction : %.6f\n" % mean_gt2recon)
        f.write("chamfer distance : %.6f\n" % chamfer_dist)
        f.write("==============================\n")

    # Plotting
    plot_graph(
        scene,
        fscore,
        dist_threshold,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
        plot_stretch,
        out_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="path to a dataset/scene directory containing X.json, X.ply, ...",
    )
    parser.add_argument(
        "--traj-path",
        type=str,
        required=True,
        help=
        "path to trajectory file. See `convert_to_logfile.py` to create this file.",
    )
    parser.add_argument(
        "--ply-path",
        type=str,
        required=True,
        help="path to reconstruction ply file",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help=
        "output directory, default: an evaluation directory is created in the directory of the ply file",
    )
    parser.add_argument(
        "--view-crop",
        type=int,
        default=0,
        help="whether view the crop pointcloud after aligned",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=-1,
        help="Number of points to uniformly sample from the mesh surface. If -1, uses default vertices and face centers.",
    )
    args = parser.parse_args()

    args.view_crop = False #  (args.view_crop > 0)
    if args.out_dir.strip() == "":
        args.out_dir = os.path.join(os.path.dirname(args.ply_path),
                                    "evaluation")

    run_evaluation(
        dataset_dir=args.dataset_dir,
        traj_path=args.traj_path,
        ply_path=args.ply_path,
        out_dir=args.out_dir,
        view_crop=args.view_crop,
        num_points=args.points
    )