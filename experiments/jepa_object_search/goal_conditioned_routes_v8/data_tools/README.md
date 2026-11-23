# 真实分支与预测选择

| 文件 | 用途 |
|---|---|
| `collect_primitives.py`、`render_labels.py` | 在 HSSD 从固定的同一相机位姿生成 12 方向观测，实际执行 9 条各 24 动作的路线，记录观测、碰撞、路程及可报告目标。 |
| `opportunity_labels.py` | 按实例匹配独立视点，先筛空间邻域，再计算可走路径距离。 |
| `replay_branches.py`、`audit_branches.py` | 重建场景并逐步复核碰撞、水平位移、三维路程，核对房屋划分、教师标签与推理输入。 |
| `learn_switch.py` | 冻结两个已训练评分器，用四间校准房屋学习何时采纳联合评分器，在另三间房屋做逐起点盲测。 |

场景配置由 `HSSD_SCENE_DATASET_CONFIG` 指向 HSSD 的 `hssd-hab.scene_dataset_config.json`。采集入口需要已获得访问许可的 HSSD、官方 ObjectNav 实例视点、已生成的补充视点目录，以及每起点的 `manifest.json` 与 `context.npz`。`--renderer` 传本目录。主评测索引中的每一行给出起点、房屋、数据划分和本地缓存位置。采集器生成的 `labels` 供训练或揭盲，`observed` 中只存当前 RGB、相机朝向与候选动作。策略一次可用的视角数随实验设置变化。

移动时每次向前 0.25 米或转向 30°。评分时看第 4、12、24 步。每个新实例在对应步数内至少一次进入可报告视点的 0.25 米可走距离才算一次。碰墙仍耗一步。在坡道上，三维路程可以超过水平步长。重复靠近同一实例只计一次。自然起点此前的报告数为零。

轻量门控先运行 [predict_switch.py](../models/predict_switch.py)，按[评分器说明](../models/README.md)只读取当前视角特征，对词表所有目标存两种策略的动作与可观测分数。再用本目录的 `learn_switch.py fit --index <校准索引> --predictions <校准预测目录> --gate <新门控文件>`，用校准房屋的真实机会差拟合带正则的线性回归。最后运行 `learn_switch.py evaluate --index <盲测索引> --predictions <盲测预测目录> --gate <已冻结门控文件> --output <新结果文件>`。每个索引条目含 `sid`、`house` 和 `labels` 路径；各阶段按物理起点、再按房屋等权汇总。运行所需图像、轨迹、场景及模型权重保存在研究服务器。
