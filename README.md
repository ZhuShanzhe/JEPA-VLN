# 用 JEPA 为多实例寻物选择探索方向

机器人按文本要求在同一栋房屋中寻找多个不同实例，例如找到三把黑色的椅子。系统以 MOST-Nav 为基线。VLM 决定何时探索、访问目标、上报或结束。研究聚焦它决定探索的时刻：给定相同的移动预算，预测哪个可达前沿更可能让机器人看清一个尚未确认、且符合查询的实例。

| 阅读入口 | 内容 |
|---|---|
| [研究问题与训练设计](experiments/jepa_object_search/RESEARCH_PLAN.md) | JEPA 接在哪一步、如何取得同一起点的多个未来，以及如何与直接打分模型比较 |
| [MOST-Nav 基线的前 16 个回合](experiments/jepa_object_search/pipeline/baseline_snapshot_16.md) | 路线覆盖、相机曝光、有效报告和重复报告的实测区别 |
| [仿真链路与对照方法](experiments/jepa_object_search/pipeline/README.md) | Habitat、VGGT-SLAM、原版导航、方向接口与评分器的验收范围 |
| [训练与开发材料](experiments/jepa_object_search/README.md) | HSSD 配对分支、Blender 受控场景、视觉预测先导实验 |
| [冻结评测与离线诊断代码](experiments/jepa_object_search/mapping_runtime/README.md) | 原版基线批量执行、按回合配对、轨迹核验和指标汇总 |

正式评测选用 MOST-Bench 的 300 个文本 ManyON 回合，类别、属性各 150 个，每回合最多执行 300 个动作。基线和方向模块共用冻结清单、建图、动作和裁判。首批统计覆盖两栋房屋。完整批次的结果以实验分支中通过验收的报告为准。[实验分支结果目录](https://github.com/ZhuShanzhe/JEPA-VLN/tree/jepa-image-nav-datasets-20261014/experiments/jepa_object_search/pipeline)
