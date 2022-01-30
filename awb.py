from typing import Tuple
from typing import List
from typing import Dict
from typing import Any
from typing import Union

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def show_color_points(points: Union[np.ndarray, List[np.ndarray]],
                      params: Union[Dict[str, Any], List[Dict[str, Any]], None] = None):
    if params is None:
        params = {"s": 40, "c": "red"}

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    if type(points) is np.ndarray:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], **params)
    else:
        if type(params) is dict:
            if "c" in params.keys():
                params.pop("c")
            params = [params] * len(points)

        for p, kw in zip(points, params):
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], **kw)

    ax.set_xlabel("B", size=15, color="black")
    ax.set_ylabel("G", size=15, color="black")
    ax.set_zlabel("R", size=15, color="black")
    plt.legend()
    plt.show()


def calc_rodrigues_rotation_formula(normal_vector: np.ndarray, theta: float) -> np.ndarray:
    sin = np.sin(theta)
    cos = np.cos(theta)
    acos = 1 - cos

    nx = normal_vector[0]
    ny = normal_vector[1]
    nz = normal_vector[2]
    return np.array([
        [cos + (nx**2)*acos, nx*ny*acos - nz*sin, nz*nx*acos + ny*sin],
        [nx*ny*acos + nz*sin, cos + (ny**2)*acos, ny*nz*acos - nx * sin],
        [nz*nx*acos - ny*sin, ny*nx*acos + nx*sin, cos + (nz**2)*acos]
    ])


def calc_rodrigues_param(average_vec: np.ndarray) -> Tuple[np.ndarray, float]:
    ideal_average_vec = np.array([127, 127, 127], dtype=np.float64)
    ideal_average_vec_norm = 127 * np.sqrt(3)

    normal_vec = np.cross(average_vec, ideal_average_vec)
    normal_vec = normal_vec / np.linalg.norm(normal_vec)
    cos = np.dot(average_vec, ideal_average_vec) / ideal_average_vec_norm / np.linalg.norm(average_vec)
    return normal_vec, np.arccos(cos)


def calc_representative_vec_ave(points: np.ndarray) -> np.ndarray:
    return points.mean(axis=0)


def cast_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.round(arr)
    arr[arr < 0] = 0
    arr[arr > 255] = 255
    return arr.astype(np.uint8)


def awb(path: str, bgr2rgb: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    img = cv2.imread(path)
    color_points = img.reshape(img.shape[0] * img.shape[1], 3)
    rep_vec = calc_representative_vec_ave(color_points)
    nv, theta = calc_rodrigues_param(rep_vec)
    rot = calc_rodrigues_rotation_formula(nv, theta)

    mapped_color_points = np.transpose(np.dot(rot, color_points.T))
    awb_img = cast_to_uint8(mapped_color_points).reshape(img.shape)

    rot_params = {"rot": rot, "normal_vector": nv, "theta": theta}

    if bgr2rgb:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(awb_img, cv2.COLOR_BGR2RGB), rot_params
    return img, awb_img, rot_params
