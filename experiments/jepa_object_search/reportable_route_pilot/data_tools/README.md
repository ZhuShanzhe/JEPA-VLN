# HSSD 可报告实例监督

一个训练起点包含若干条候选路线。模拟器从相同身体位姿出发，各执行 4、12 或 24 个动作。目标值是各路线首次进入可报告范围的不同目标实例数。计数时扣除出发时已在范围内的实例和此前正式报告过的实例。曾经看见、却尚未报告和接近的实例可以计分。

## 官方视点与自建视点

官方六类读取 HSSD ObjectNav v0.2.5 任务包中实例的 `view_points`。扩展家居类别使用 HSSD 的原始实例与语义资产，在独立于待评分路线的世界坐标网格上生成视点。两种目录分开计算指标。扩展目录的十类为壁饰、画框、门、镜子、钟、灯、柜子、搁板、篮子和凳子。壁饰数据没有“彩色风景画”属性，门数据没有“白门”属性。

生成器需要可用的 Habitat-Sim、NumPy、Magnum、场景资产和可渲染设备。将以下参数替换为 HSSD 数据目录、房屋 ID 和输出目录。

```bash
python derive_viewpoints.py --raw HSSD_ROOT --house HOUSE_ID --output generated_goals
```

程序将导航网格设置为机器人高 1.4 米、半径 0.18 米，并纳入静态物体。独立网格的间距为 0.4 米，在每件物体水平距离 0.45–2 米内寻找合法站位。可导航位置通过实例语义图和深度图检查。相机离地 1.25 米，水平视场 90°，视图分辨率 128×128。目标需占至少 48 个像素，其中至少 32 个像素深度在 2 米内。每个房屋的输出文件记录有效视点、相机姿态及未获得有效视点的实例。

## 为一组候选路线标注

`opportunity_labels.py` 的 `ReportViewpoints` 读取 Habitat 的 `pathfinder` 和实例目录。可先用欧氏距离排除不可能的候选，最终正例要求到某个合法视点的可行走最短距离不超过 0.25 米。输入的世界位置须来自在同一张导航网格上执行的行动。派生目录需先剔除没有可用视点的实例。

```python
import numpy as np
from opportunity_labels import ReportViewpoints, new_instance_counts

teacher = ReportViewpoints(sim.pathfinder, goals)
start = teacher.eligible(start_position)
hit = np.asarray([[teacher.eligible(p) for p in route] for route in branch_positions])
target_mask = [g["object_category"] == target_category for g in goals]
counts = new_instance_counts(hit, start, teacher.ids, target_mask, reported_ids)
```

`reported_ids` 是出发前已报告的实例集合。当前自然分支数据的该集合为空；另有回到同一起点的受控模拟报告用于核实去重。目标物体 ID、未来相机画面、实例掩码和可报告视点仅用于教师标注与评测。

## 核验

`audit_labels.py` 读取两套标签和它们引用的分支几何缓存，复算每个起点、时长、物类和分支的得分。`replay_catalog.py` 对生成的视点重新载入房屋，重排实例的渲染 ID，再检查站位、相机高度、物体像素数和近处像素数。两项核验使用原始授权资产和本地缓存。[房屋划分与像素复核汇总](../dataset_report.json)
