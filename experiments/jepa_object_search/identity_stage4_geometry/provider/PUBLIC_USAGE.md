# 将局部视觉重建接入实例记忆

输入为登记物体的关键帧、已观察的八个视角和当前 RGB。局部 VGGT 一起估计十帧的深度、内参和相机位姿。已有地图中的两个历史关键帧用于把局部重建放回实例记忆使用的坐标系。第一帧定标，第二帧独立核对三维位置和朝向。

## 接口

- `optics.skew_view` 读取已提交地图快照及独立获得的米制尺度，保留相机内参中的水平剪切量。快照含相机到世界的姿态 `pose`、按 8 像素稀疏采样的世界点 `points`、对应图像行 `rows` 和 `frame_id`。
- `shared_anchor_pair.split_camera_height_scales` 使用原导航器的地板估计函数与标定安装高度，从两组互不重叠的过去观测各估计一次尺度。两组相差超过 15% 时暂缓匹配。
- `shared_anchor_pair.verify_pair` 接收同一像素坐标系的登记框和当前候选框、局部重建、原地图中的登记视图与共享历史帧，以及冻结的身份判别头。返回 `result` 和 `reason`。通过判别后还返回候选序号与分数。调用方负责决定是否提交报告。

```python
from optics import skew_view
from shared_anchor_pair import split_camera_height_scales, verify_pair

# 顺序固定为登记帧、八张过去图、当前图。RGB 为同分辨率 uint8。
# depth、K、c2w 由这十张图的同一次局部重建产生，使用 OpenCV 相机轴。
floor_scales = split_camera_height_scales(
    depth[1:9], K[1:9], c2w[1:9], navigator, height_m=camera_mount_height,
)
source_view = skew_view(source_snapshot, online_visual_scale)
anchors = [
    (frame_id, image_index, native_rgb, skew_view(snapshot, online_visual_scale))
    for frame_id, image_index, native_rgb, snapshot in shared_past_frames
]
decision = verify_pair(
    images, depth, K, c2w, source_view, source_xyxy, live_xyxy,
    anchors, floor_scales, frozen_identity_verifier,
)
```

`image_index` 为 1 到 8。共享帧的原图须能和该索引的实际 RGB 对齐。正常调用由接口从 RGB 提取双向匹配并核验重投影。若重用 `verified_image_edges` 的缓存，缓存必须绑定同一组图像、深度、内参和位姿。所有帧须在当前决策时已经观测到。

## 回放检查

最近八步的策略在两条轨迹共 55 个请求中恢复了 1 次可评分的实例配对。将最近两个有效几何锚纳入相同的八图预算后，可进入局部重建的请求由 18 次增至 46 次，最终仍只有这一次能评分。该请求是方法开发样本，余下 54 次均暂缓。首例选框覆盖已登记时钟可见像素的 81.77%，框内目标像素占 21.26%。框中心位于背景。闭环任务成绩与这些离线身份判定应分别统计。
