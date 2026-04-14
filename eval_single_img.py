#!/usr/bin/env python3
"""
RoomFormer 单张图片推理脚本
功能：输入单张原始图片，输出语义丰富的楼层平面图结果
"""

import argparse
import math
import os
import cv2
import numpy as np
from PIL import Image
import torch
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.ops import unary_union
from pathlib import Path

from models import build_model
from util.plot_utils import plot_semantic_rich_floorplan

try:
    import ezdxf
except ImportError:
    ezdxf = None


def get_args_parser():
    """
    命令行参数解析器
    """
    parser = argparse.ArgumentParser('RoomFormer 单张图片推理', add_help=False)

    # 输入输出参数
    parser.add_argument('--input_img', required=True, type=str,
                        help='输入图片路径')
    parser.add_argument('--output_img', required=True, type=str,
                        help='输出结果图片路径')
    parser.add_argument('--output_dxf', default='', type=str,
                        help='可选，输出 DXF 路径')
    parser.add_argument('--checkpoint', default='checkpoints/roomformer_stru3d_semantic_rich.pth',
                        type=str, help='模型检查点路径')
    parser.add_argument('--wall_thickness', default=4.0, type=float,
                        help='DXF 中墙线的可视化厚度，单位与图像像素一致')

    # 模型参数（语义丰富模型默认配置）
    parser.add_argument('--num_queries', default=2800, type=int,
                        help="角点查询总数")
    parser.add_argument('--num_polys', default=70, type=int,
                        help="最大房间数量")
    parser.add_argument('--semantic_classes', default=19, type=int,
                        help="语义类别数（19 = 16种房间 + 门 + 窗 + 空）")

    # Backbone 参数
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="卷积骨干网络名称")
    parser.add_argument('--lr_backbone', default=0, type=float)
    parser.add_argument('--dilation', action='store_true',
                        help="如果为true，在最后一个卷积块中用空洞卷积替换步长")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="在图像特征上使用的位置嵌入类型")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="位置 / 尺寸 * 缩放")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='特征层级数')

    # Transformer 参数
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Transformer 编码器层数")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Transformer 解码器层数")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Transformer 前馈层的中间维度")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="嵌入的维度（Transformer 的维度）")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Transformer 中的 Dropout")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Transformer 注意力头数")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    parser.add_argument('--query_pos_type', default='sine', type=str, choices=('static', 'sine', 'none'),
                        help="解码器中查询位置的类型")
    parser.add_argument('--with_poly_refine', default=True, action='store_true',
                        help="迭代细化参考点")
    parser.add_argument('--masked_attn', default=False, action='store_true',
                        help="如果为true，一个房间中的查询将不被允许关注其他房间")

    # 辅助损失
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_true',
                        help="禁用辅助解码损失")

    # 设备参数
    parser.add_argument('--device', default='cuda',
                        help='用于训练/测试的设备')
    parser.add_argument('--seed', default=42, type=int)

    return parser


def preprocess_image(image_path, device):
    """
    预处理单张图片

    Args:
        image_path: 输入图片路径
        device: 计算设备

    Returns:
        预处理后的张量，形状为 [1, H, W]
    """
    # 读取图片
    img = np.array(Image.open(image_path))

    # 确保图片是 256x256 的灰度图
    if len(img.shape) == 3:
        # 转换为灰度图
        img = np.mean(img, axis=2)

    # 归一化到 [0, 1] 并转换为张量
    img_tensor = (1 / 255) * torch.as_tensor(np.ascontiguousarray(np.expand_dims(img, 0)))

    # 移动到设备
    img_tensor = img_tensor.to(device)

    return img_tensor


def postprocess_outputs(outputs):
    """
    后处理模型输出

    Args:
        outputs: 模型输出字典

    Returns:
        room_polys: 房间多边形列表
        room_types: 房间类型列表
        window_doors: 门窗线列表
        window_doors_types: 门窗类型列表
    """
    pred_logits = outputs['pred_logits']
    pred_corners = outputs['pred_coords']
    fg_mask = torch.sigmoid(pred_logits) > 0.5  # 选择有效角点

    # 获取房间语义标签
    prob = torch.nn.functional.softmax(outputs['pred_room_logits'], -1)
    _, pred_room_label = prob[..., :-1].max(-1)

    # 处理单张图片（batch size = 1）
    i = 0
    fg_mask_per_scene = fg_mask[i]
    pred_corners_per_scene = pred_corners[i]
    pred_room_label_per_scene = pred_room_label[i].cpu().numpy()

    room_polys = []
    room_types = []
    window_doors = []
    window_doors_types = []

    # 处理每个房间
    for j in range(fg_mask_per_scene.shape[0]):
        fg_mask_per_room = fg_mask_per_scene[j]
        pred_corners_per_room = pred_corners_per_scene[j]
        valid_corners_per_room = pred_corners_per_room[fg_mask_per_room]

        if len(valid_corners_per_room) > 0:
            corners = (valid_corners_per_room * 255).cpu().numpy()
            corners = np.around(corners).astype(np.int32)

            # 普通房间
            if pred_room_label_per_scene[j] not in [16, 17]:
                if len(corners) >= 4 and Polygon(corners).area >= 100:
                    room_polys.append(corners)
                    room_types.append(pred_room_label_per_scene[j])
            # 门/窗
            elif len(corners) == 2:
                window_doors.append(corners)
                window_doors_types.append(pred_room_label_per_scene[j])

    return room_polys, room_types, window_doors, window_doors_types


def _remove_consecutive_duplicate_points(points):
    """
    删除连续重复点，避免生成无效多边形。
    """
    if len(points) == 0:
        return points

    dedup_points = [points[0]]
    for point in points[1:]:
        if not np.array_equal(point, dedup_points[-1]):
            dedup_points.append(point)

    if len(dedup_points) > 1 and np.array_equal(dedup_points[0], dedup_points[-1]):
        dedup_points.pop()

    return np.asarray(dedup_points, dtype=np.int32)


def orthogonalize_polygon(poly):
    """
    将预测多边形做轻量正交化。

    规则：
    1. 每条边按照主方向吸附到水平或垂直。
    2. 闭合边再做一次吸附，尽量保持矩形/正交房间的外观。
    """
    poly = np.asarray(poly, dtype=np.float32)
    if len(poly) < 3:
        return np.asarray(poly, dtype=np.int32)

    ortho = poly.copy()

    for i in range(len(ortho) - 1):
        dx = ortho[i + 1, 0] - ortho[i, 0]
        dy = ortho[i + 1, 1] - ortho[i, 1]
        if abs(dx) >= abs(dy):
            ortho[i + 1, 1] = ortho[i, 1]
        else:
            ortho[i + 1, 0] = ortho[i, 0]

    dx = ortho[0, 0] - ortho[-1, 0]
    dy = ortho[0, 1] - ortho[-1, 1]
    if abs(dx) >= abs(dy):
        snap_y = round((ortho[0, 1] + ortho[-1, 1]) / 2.0)
        ortho[0, 1] = snap_y
        ortho[-1, 1] = snap_y
    else:
        snap_x = round((ortho[0, 0] + ortho[-1, 0]) / 2.0)
        ortho[0, 0] = snap_x
        ortho[-1, 0] = snap_x

    ortho = np.round(ortho).astype(np.int32)
    ortho = _remove_consecutive_duplicate_points(ortho)

    if len(ortho) < 3:
        return np.asarray(poly, dtype=np.int32)

    try:
        if Polygon(ortho).area <= 0:
            return np.asarray(poly, dtype=np.int32)
    except Exception:
        return np.asarray(poly, dtype=np.int32)

    return ortho


def orthogonalize_line(line):
    """
    将门窗端点吸附到水平或垂直方向。
    """
    line = np.asarray(line, dtype=np.float32)
    if len(line) != 2:
        return np.asarray(line, dtype=np.int32)

    ortho = line.copy()
    dx = ortho[1, 0] - ortho[0, 0]
    dy = ortho[1, 1] - ortho[0, 1]
    if abs(dx) >= abs(dy):
        ortho[1, 1] = ortho[0, 1]
    else:
        ortho[1, 0] = ortho[0, 0]
    return np.round(ortho).astype(np.int32)


def _flip_y(points, canvas_size=256):
    """
    将图像坐标系转换为 CAD 常用的 y 轴向上坐标系。
    """
    flipped = np.asarray(points, dtype=np.float32).copy()
    flipped[:, 1] = (canvas_size - 1) - flipped[:, 1]
    return flipped


def _iter_polygon_geometries(geom):
    """
    从 shapely 几何体中提取 Polygon，兼容 MultiPolygon / GeometryCollection。
    """
    if geom.is_empty:
        return

    if isinstance(geom, Polygon):
        yield geom
        return

    if isinstance(geom, MultiPolygon):
        for sub_geom in geom.geoms:
            yield from _iter_polygon_geometries(sub_geom)
        return

    if isinstance(geom, GeometryCollection):
        for sub_geom in geom.geoms:
            yield from _iter_polygon_geometries(sub_geom)


def _point_in_any_room(point, room_polygons):
    """
    判断一个点是否位于任意房间内部。
    """
    for room_polygon in room_polygons:
        if room_polygon.buffer(1e-6).contains(point):
            return True
    return False


def _sort_line_endpoints(line):
    """
    为门窗线段固定端点顺序，便于后续稳定生成圆弧。
    """
    line = np.asarray(line, dtype=np.float32)
    if len(line) != 2:
        return line

    p0, p1 = line[0], line[1]
    dx = abs(p1[0] - p0[0])
    dy = abs(p1[1] - p0[1])

    if dx >= dy:
        return np.asarray([p0, p1], dtype=np.float32) if p0[0] <= p1[0] else np.asarray([p1, p0], dtype=np.float32)
    return np.asarray([p0, p1], dtype=np.float32) if p0[1] <= p1[1] else np.asarray([p1, p0], dtype=np.float32)


def _compute_door_arc(line, room_polygons, wall_thickness):
    """
    根据门洞线段生成门扇线和开启圆弧参数。

    返回：
    - hinge: 合页点
    - leaf_end: 开启后门扇终点
    - radius: 圆弧半径
    - start_angle / end_angle: DXF ARC 所需角度
    """
    line = _sort_line_endpoints(line)
    p0 = np.asarray(line[0], dtype=np.float32)
    p1 = np.asarray(line[1], dtype=np.float32)
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return None

    direction_unit = direction / length
    normal_left = np.asarray([-direction_unit[1], direction_unit[0]], dtype=np.float32)
    normal_right = -normal_left
    test_offset = max(wall_thickness * 0.75, 3.0)
    mid = (p0 + p1) / 2.0

    inside_left = _point_in_any_room(Point(mid + normal_left * test_offset), room_polygons)
    inside_right = _point_in_any_room(Point(mid + normal_right * test_offset), room_polygons)

    if inside_left and not inside_right:
        swing_normal = normal_left
    elif inside_right and not inside_left:
        swing_normal = normal_right
    else:
        swing_normal = normal_left

    hinge = p0
    leaf_end = hinge + swing_normal * length
    start_angle = math.degrees(math.atan2(direction[1], direction[0]))
    end_vector = leaf_end - hinge
    end_angle = math.degrees(math.atan2(end_vector[1], end_vector[0]))

    return {
        "hinge": hinge,
        "leaf_end": leaf_end,
        "radius": length,
        "start_angle": start_angle,
        "end_angle": end_angle,
        "closed_end": p1,
    }


def _build_wall_contours_from_rooms(room_polygons, wall_thickness, canvas_size=256, render_scale=4):
    """
    用纯几何方式生成统一墙厚的双线轮廓。

    逻辑：
    1. 对每个房间边界做等厚 buffer，得到墙带
    2. 对整体墙带做一次小尺度几何闭合，补齐微小缝隙
    3. 直接从几何体提取外轮廓/内轮廓，避免栅格化带来的厚度漂移
    """
    half_wall = max(wall_thickness / 2.0, 0.5)
    contour_groups = []
    if not room_polygons:
        return contour_groups

    wall_geometries = [
        room_polygon.boundary.buffer(half_wall, join_style=2, cap_style=2)
        for room_polygon in room_polygons
    ]
    wall_union = unary_union(wall_geometries)

    # 用几何闭合补齐很小的断缝，同时尽量保持原始墙厚。
    close_eps = max(0.75, wall_thickness * 0.2)
    wall_union = wall_union.buffer(close_eps, join_style=2).buffer(-close_eps, join_style=2)

    for wall_polygon in _iter_polygon_geometries(wall_union):
        shell = _orthogonalize_ring(np.asarray(wall_polygon.exterior.coords[:-1], dtype=np.float32))
        if len(shell) < 4:
            continue

        holes = []
        for interior in wall_polygon.interiors:
            hole = _orthogonalize_ring(np.asarray(interior.coords[:-1], dtype=np.float32))
            if len(hole) >= 4:
                holes.append(hole)

        contour_groups.append({
            "shell": shell,
            "holes": holes,
        })

    return contour_groups


def _cluster_axis_values(values, tol):
    """
    将相近的坐标值聚成一组，返回每个原值对应的吸附值。
    """
    if not values:
        return {}

    sorted_values = sorted(float(v) for v in values)
    groups = [[sorted_values[0]]]

    for value in sorted_values[1:]:
        if abs(value - groups[-1][-1]) <= tol:
            groups[-1].append(value)
        else:
            groups.append([value])

    mapping = {}
    for group in groups:
        snapped = float(np.median(group))
        for value in group:
            mapping[value] = snapped

    return mapping


def _snap_polygons_to_global_axes(polygons, tol):
    """
    把所有房间角点吸附到全局共享的 x/y 轴线上，减少局部墙厚漂移。
    """
    if not polygons:
        return []

    x_values = []
    y_values = []
    for polygon in polygons:
        x_values.extend(polygon[:, 0].tolist())
        y_values.extend(polygon[:, 1].tolist())

    x_mapping = _cluster_axis_values(x_values, tol)
    y_mapping = _cluster_axis_values(y_values, tol)

    snapped_polygons = []
    for polygon in polygons:
        snapped = polygon.astype(np.float32).copy()
        for idx in range(len(snapped)):
            snapped[idx, 0] = x_mapping.get(float(snapped[idx, 0]), float(snapped[idx, 0]))
            snapped[idx, 1] = y_mapping.get(float(snapped[idx, 1]), float(snapped[idx, 1]))

        snapped = _remove_consecutive_duplicate_points(np.round(snapped).astype(np.int32))
        if len(snapped) >= 3:
            snapped_polygons.append(snapped.astype(np.float32))

    return snapped_polygons


def _orthogonalize_ring(points):
    """
    将轮廓环压缩为严格水平/垂直的折线，同时尽量保持几何尺寸。
    """
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    dedup_points = [points[0]]
    for point in points[1:]:
        if not np.allclose(point, dedup_points[-1]):
            dedup_points.append(point)

    if len(dedup_points) > 1 and np.allclose(dedup_points[0], dedup_points[-1]):
        dedup_points.pop()

    if len(dedup_points) < 2:
        return np.asarray(dedup_points, dtype=np.float32)

    ortho_points = [np.asarray(dedup_points[0], dtype=np.float32)]
    prev_dir = None

    for idx in range(1, len(dedup_points) + 1):
        next_point = np.asarray(dedup_points[idx % len(dedup_points)], dtype=np.float32)
        curr_point = ortho_points[-1].copy()

        dx = next_point[0] - curr_point[0]
        dy = next_point[1] - curr_point[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue

        curr_dir = "h" if abs(dx) >= abs(dy) else "v"
        snapped_point = next_point.copy()
        if curr_dir == "h":
            snapped_point[1] = curr_point[1]
        else:
            snapped_point[0] = curr_point[0]

        if prev_dir is None:
            ortho_points.append(snapped_point)
            prev_dir = curr_dir
            continue

        if curr_dir == prev_dir:
            ortho_points[-1] = snapped_point
        else:
            ortho_points.append(snapped_point)
            prev_dir = curr_dir

    if len(ortho_points) > 1 and np.allclose(ortho_points[0], ortho_points[-1]):
        ortho_points.pop()

    cleaned_points = []
    num_points = len(ortho_points)
    for idx in range(num_points):
        prev_point = ortho_points[idx - 1]
        curr_point = ortho_points[idx]
        next_point = ortho_points[(idx + 1) % num_points]
        vec1 = curr_point - prev_point
        vec2 = next_point - curr_point
        if (abs(vec1[0]) < 1e-6 and abs(vec2[0]) < 1e-6) or (abs(vec1[1]) < 1e-6 and abs(vec2[1]) < 1e-6):
            continue
        cleaned_points.append(curr_point)

    if len(cleaned_points) < 4:
        cleaned_points = ortho_points

    return np.round(np.asarray(cleaned_points, dtype=np.float32), 3)


def process_floorplan_for_dxf(room_polys, room_types, window_doors, window_doors_types, wall_thickness, canvas_size=256):
    """
    为 DXF 导出准备几何：
    1. 房间 polygon 正交化
    2. 门窗线段正交化
    3. 构建共享墙体的双线轮廓
    4. 在墙体上扣除门窗开口
    """
    processed = {
        "rooms": [],
        "doors": [],
        "windows": [],
        "wall_contours": [],
    }

    room_polygon_points = []
    room_polygon_types = []

    for poly, room_type in zip(room_polys, room_types):
        ortho_poly = orthogonalize_polygon(poly)
        if len(ortho_poly) < 3:
            continue

        cad_poly = _flip_y(ortho_poly, canvas_size=canvas_size)
        room_polygon_points.append(cad_poly.astype(np.float32))
        room_polygon_types.append(int(room_type))

    snap_tol = max(1.0, wall_thickness * 0.75)
    snapped_room_polygon_points = _snap_polygons_to_global_axes(room_polygon_points, tol=snap_tol)

    room_polygons = []
    processed["rooms"] = []
    for snapped_poly, room_type in zip(snapped_room_polygon_points, room_polygon_types):
        room_polygon = Polygon(snapped_poly)
        if (not room_polygon.is_valid) or room_polygon.area <= 0:
            continue

        room_polygons.append(room_polygon)

        processed["rooms"].append({
            "polygon": snapped_poly,
            "room_type": int(room_type),
        })

    if room_polygons:
        processed["wall_contours"] = _build_wall_contours_from_rooms(
            room_polygons=room_polygons,
            wall_thickness=wall_thickness,
            canvas_size=canvas_size,
        )

    return processed


def export_floorplan_to_dxf(processed_floorplan, dxf_path):
    """
    将处理后的平面图导出为 DXF。

    墙体使用“带厚度的线框轮廓”表示：
    不是设置 CAD 线宽，而是直接输出墙体环带的外轮廓/内轮廓。
    """
    if ezdxf is None:
        raise ImportError("未安装 ezdxf，请先执行: pip install ezdxf")

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    layer_specs = {
        "WALL": {"color": 7},
        "DOOR": {"color": 4},
        "WINDOW": {"color": 2},
    }
    for layer_name, attrs in layer_specs.items():
        if layer_name not in doc.layers:
            doc.layers.add(name=layer_name, color=attrs["color"])

    for wall_group in processed_floorplan["wall_contours"]:
        shell = np.asarray(wall_group["shell"], dtype=np.float64)
        if len(shell) >= 2:
            msp.add_lwpolyline(shell.tolist(), close=True, dxfattribs={"layer": "WALL"})

        for hole in wall_group["holes"]:
            hole = np.asarray(hole, dtype=np.float64)
            if len(hole) >= 2:
                msp.add_lwpolyline(hole.tolist(), close=True, dxfattribs={"layer": "WALL"})

    for door in processed_floorplan["doors"]:
        arc = door.get("arc")
        if arc is not None:
            msp.add_line(
                tuple(np.asarray(arc["hinge"], dtype=np.float64)),
                tuple(np.asarray(arc["leaf_end"], dtype=np.float64)),
                dxfattribs={"layer": "DOOR"},
            )
            msp.add_arc(
                center=tuple(np.asarray(arc["hinge"], dtype=np.float64)),
                radius=float(arc["radius"]),
                start_angle=float(arc["start_angle"]),
                end_angle=float(arc["end_angle"]),
                dxfattribs={"layer": "DOOR"},
            )
        else:
            msp.add_line(
                tuple(door["line"][0]),
                tuple(door["line"][1]),
                dxfattribs={"layer": "DOOR"},
            )

    for window in processed_floorplan["windows"]:
        window_outline = LineString(window["line"]).buffer(1.0, cap_style=2)
        for window_polygon in _iter_polygon_geometries(window_outline):
            exterior = np.asarray(window_polygon.exterior.coords[:-1], dtype=np.float64)
            if len(exterior) >= 2:
                msp.add_lwpolyline(exterior.tolist(), close=True, dxfattribs={"layer": "WINDOW"})

    doc.saveas(dxf_path)


def main(args):
    """
    主函数：加载模型、推理、保存结果
    """
    device = torch.device(args.device)

    # 固定随机种子以保证可重复性
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 构建模型
    print("正在构建模型...")
    model = build_model(args, train=False)
    model.to(device)

    # 加载检查点
    print(f"正在加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]

    if len(missing_keys) > 0:
        print(f'缺失的键: {missing_keys}')
    if len(unexpected_keys) > 0:
        print(f'多余的键: {unexpected_keys}')

    model.eval()

    # 预处理图片
    print(f"正在预处理图片: {args.input_img}")
    img_tensor = preprocess_image(args.input_img, device)

    # 推理
    print("正在进行推理...")
    with torch.no_grad():
        outputs = model([img_tensor])  # 模型期望列表输入

    # 后处理
    print("正在后处理结果...")
    room_polys, room_types, window_doors, window_doors_types = postprocess_outputs(outputs)

    # 准备可视化数据
    pred_sem_rich = []

    # 添加房间
    for j in range(len(room_polys)):
        temp_poly = room_polys[j]
        temp_poly_flip_y = temp_poly.copy()
        temp_poly_flip_y[:, 1] = 255 - temp_poly_flip_y[:, 1]  # 翻转 y 轴
        pred_sem_rich.append([temp_poly_flip_y, room_types[j]])

    # 添加门窗
    for j in range(len(window_doors)):
        temp_line = window_doors[j]
        temp_line_flip_y = temp_line.copy()
        temp_line_flip_y[:, 1] = 255 - temp_line_flip_y[:, 1]  # 翻转 y 轴
        pred_sem_rich.append([temp_line_flip_y, window_doors_types[j]])

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_img)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 生成并保存结果图
    print(f"正在保存结果到: {args.output_img}")
    plot_semantic_rich_floorplan(pred_sem_rich, args.output_img)

    if args.output_dxf:
        print(f"正在生成 DXF: {args.output_dxf}")
        processed_floorplan = process_floorplan_for_dxf(
            room_polys=room_polys,
            room_types=room_types,
            window_doors=window_doors,
            window_doors_types=window_doors_types,
            wall_thickness=args.wall_thickness,
        )
        export_floorplan_to_dxf(processed_floorplan, args.output_dxf)

    print("推理完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RoomFormer 单张图片推理脚本', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
