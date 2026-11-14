# 多实例寻物中的前沿选择

MOST-Nav 由 VLM 决定当前要探索、访问已知实例、报告还是结束。探索时，对同一组合法前沿预测在短程移动后可能获得的视觉证据，再根据查询和已报告实例选方向。原系统决定其他动作并执行移动。SCOPE、3D-Mem 和 MTU3D 作为外部整机对照。

| 资源 | 验证范围 |
|---|---|
| HSSD | 官方 ObjectNav v0.2.5 有 122 个训练场景、120,552 回合，验证集有 42 个场景、1,248 回合。分支先导集另取 12 个训练房屋、4 个开发房屋，共 765 条 24 步分支。 |
| MOST-Bench | 作者加载器读通 36 个房屋的 750 条任务。公开条件名 `category_many`、`attribute_many` 各 150 条构成主评测子集。清单 SHA-256 为 `48f0f21020b3038121efd48e4f6dc880e47d3d1420c3a54745b3b55301d5cd3d`。环境采用 Habitat-Lab 和 Habitat-Sim 0.3.3。 |
| Blender / Habitat | 两个同类实例位于遮挡两侧。两个等预算分支从相同视角出发，各用 19 个动作取得不同的目标证据。报告图像与相机位置来自实际往返轨迹，报告事件由受控实验构造。 |
| SCOPE | 真实 HSSD 图像通过共享感知前端；结构化潜力图在给定评分与前沿下运行。 |
| 3D-Mem | 真实 HSSD RGB-D 帧通过原版 TSDF 融合，更新 7,211 个体素。公开 MOST 适配器提供多次报告的回合协议。 |
| MTU3D | 在一个 HSSD 训练房屋输入 12 帧实际 RGB-D、相机位姿和两个可达前沿。两阶段分别完整载入 565、392 项状态，对“电视”目标的 10 个候选打分，输出一个移动路点。这是一次原版决策的工程验收。 |

## 模块

在自建房间，原版 MOST 评测入口已运行两个脚本控制的完整回合，各移动 5 米。两次正确报告均获认可；增加一次重复报告和一次距离过远的报告后，原评分器记为 2 个真阳性、2 个假阳性和 1 次重复。作者公开包 177 个文件的内容与下载包一致。

HSSD 的 144 个训练起点中有 36 个起点可观察到两个同类实例，3 个起点满足两个方向分别显露不同实例、可真实到达且无碰撞的采样条件。在其中两栋训练房屋记录了 6 段往返历史，共得到 28 个候选与历史配对标签。观察及移动来自仿真，报告记账是受控构造。另有 Blender 颜色交叉场景，保持四条路线的几何和动作一致，通过更换查询或可见线索翻转正确方向。纯距离规则答对 2/4，同色提示规则答对 4/4。

| 文件 | 用途 |
|---|---|
| `ops/frontier_routing.py`、`ops/check_frontier_routing.py` | 回传原控制器的前沿编号、旁路与接管检查 |
| `frontier_jepa/policy.py`、`frontier_jepa/vision.py` | 读取过去 RGB、候选动作和位置，预测并打分 |
| `frontier_jepa/run_pilot.py`、`frontier_jepa/ablate_readout.py` | HSSD 成对先导训练与三种子四组消融 |
| `frontier_jepa/memory.py`、`frontier_jepa/fit_report_fixture.py` | 仅凭已观测的视觉报告与位置比较路线，在受控单房验证接口 |
| `frontier_scenes/tools/` | 生成 Blender 几何，导入 Habitat，记录相同起点的两条路线和报告历史 |

实验运行目录按本目录结构布置。`a_exploration/paired_branches_v2/` 为本地 HSSD 分支缓存，`models/vjepa2/` 为主干权重，`src/vjepa2/` 为官方模型源码。训练和验收命令见各目录 README。

## 判读

SCOPE 使用源码提交 `46f1e1f`，其潜力图预检使用两个固定的 VLM 测试响应。MTU3D 使用提交 `c10335c` 和作者 GOAT 权重；同一 HSSD 起点按该模型的导航网格吸附约 0.11 米。MTU3D 的 42° 视场、1.31 米相机和 12 次离散转向用于跨环境工程验收。两阶段校验使用的 SHA-256 分别为 `8d064237059b673eb0d93594fcc9d1f927afa4e8bd4fbb9ec0199c87bef42de0` 和 `7cee579a00f85b802ddaafc4c649f56228b6eb8090b6ff5e06c3ff9470500162`。

HSSD 的代理目标是新增可见性。开发集 48 个起点中只有 2 个起点的不同候选产生不同的二元收益，均在同一栋房屋。三个种子中，加入未来监督后，无论决策时从当前条件还是预测未来读取，均在 48 个起点中选中 1 次正收益分支。单房双实例任务在调换报告历史时会改变方向，纯距离规则也能解出。两类数据是工程验收和消融流程的验证材料。

正式多实例闭环需取得 MOST-Nav 原版配置及 HM3D v0.2 场景。主基线与改进版在相同传感器、动作预算和报告协议下按回合配对，分别报告配额完成率、实例召回率、精确率、F1、重复报告率及行程。外部方法注明其额外传感器、定位输入和原生步长。[MOST-Bench](https://most-bench.netlify.app/) · [HSSD](https://github.com/3dlg-hcvc/hssd) · [SCOPE](https://mrwangyou.github.io/SCOPE/) · [3D-Mem](https://github.com/UMass-Embodied-AGI/3D-Mem) · [MTU3D](https://mtu3d.github.io/)

[早期导航与主动核验资源](earlier_navigation_resources.md)
