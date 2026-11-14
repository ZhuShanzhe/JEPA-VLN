# 多实例寻物实验流程

Habitat 提供观测并执行动作，VGGT-SLAM 从连续图像估计相机位姿和三维地图。MOST-Nav 用地图生成可达探索前沿，由 VLM 选择探索、访问已知实例、报告或结束。方向选择模块只在 VLM 要求探索且至少有两个合法前沿时接入。原路径跟随器执行选中的方向，MOST-Bench 裁判核对实例报告。

## 真实场景验收

| 环节 | 工程样本 | 实际观测 |
|---|---|---|
| 三维建图 | HM3D，30 次相机观测、29 个动作 | VGGT-SLAM 建立 2 张子图，输出 27 个相机位姿；从地图读回 100,000 个有限三维点。机器人相对起点最大位移 2.21 米。 |
| 语义地图 | HM3D 导航 | 图像描述、SAM 区域分割、区域选择均有实际调用记录。BGE-M3 的单条文本嵌入为 1,024 维。回环检测已启用，轨迹定位误差和回环识别率尚待专门评测。 |
| 原版导航 | 冻结集外的 AllON 回合，无干预观测器 | 300 个策略动作逐步匹配模拟器记录。13 次 VLM 决策含 1 次扫描、12 次合法前沿选择。 |
| 原版多实例导航 | 非冻结双实例 ManyON 开发任务，直接运行作者的 NavAgent | 走完 300 步、31.04 米。计分器记录 0 个正确报告、1 个误报。此任务复用了 AllON 房屋、起点和目标几何。 |
| 更换移动方向 | 同一决策点的 4 个合法前沿 | VLM 选 f0，工程回调改选 f1，路径跟随器执行四次转向和一次前进。模拟器确认位移 0.25 米。工程回调只验证控制接口。 |
| 分支观测采集 | HSSD 训练房屋，同起点的左右两条 5 动作分支 | 两次初始 RGB 完全相同，实际终点 RGB 不同。当前观测和候选动作与执行后图像分别存储。 |
| 属性任务输入 | 全部 150 条属性指令 | 1,190 个目标引用均带查询所需的属性标注。受控核对中，“寻找黑白画”的裁判排除了同类彩色风景画的观察点。 |
| 实例计分 | Blender / Habitat 双实例受控任务 | 两回合共 4 次正确报告、2 次误报，其中 1 次重复报告。 |

## 外部方法

| 方法 | 已执行的链路 | 运行范围 |
|---|---|---|
| SCOPE | 感知、潜力图、快照或前沿选择、航点移动、裁判计分 | 原生 GOAT 一个回合的 10 个子任务均有结果。在共享双实例开发任务中实际选择前沿并移动 0.98 米，开发预算为 1 个航点步。 |
| 3D-Mem | 多视角 RGB-D、TSDF 地图、快照或前沿选择、执行、计分 | 在同一开发任务中连续选了两个前沿，实际路径 2.05 米，并按 2 个航点步的开发预算结算。另将受控上报交给其适配裁判，6 次判定均与主裁判一致。 |
| MTU3D | 预训练模型选择对象或前沿、移动、报告、计分 | 原生 GOAT 一个回合完成 6 个子目标的评测。在同一双实例开发任务中走了 226 步、18.30 米，2 次上报分别为正确报告和重复报告。 |

主评测采用 36 栋 HM3D 房屋上的 300 个冻结 ManyON 文本回合，类别、属性各 150 个。MOST-Nav 基线与后续方向选择模块共用起点、传感器、动作预算和裁判。当前主链路用 GPS 估计尺度，决策和图像描述均固定为 `gpt-4o-2024-11-20`，作者已停用的旧 pointing 后端关闭。外部方法的航点步、RGB-D、位姿等输入和动作单位按各自适配协议记录。上述开发样本用于验收，冻结集性能及 JEPA 增益待评测。

## 服务器运行入口

项目根目录为 `/root/autodl-tmp/jepa_object_search`。下列命令从该目录运行，运行标识须唯一。第一条在独立终端启动建图服务，其余命令通过本机端口连接。接口测试会使用已配置的视觉模型服务。

```bash
bash mapping_runtime/harness/run_pipeline.sh server map_service_$(date +%s)
bash mapping_runtime/harness/run_pipeline.sh map geometry_$(date +%s)
bash mapping_runtime/harness/run_pipeline.sh native-many-check native_many_$(date +%s)
run=direction_$(date +%s)
MOST_EXEC_SCORER=execute_frontier:engineering_alternate bash mapping_runtime/harness/run_pipeline.sh execution-check "$run"
/root/miniconda3/bin/python mapping_runtime/harness/verify_branch.py "mapping_runtime/runs/$run"
```

| 服务器文件 | 用途 |
|---|---|
| `mapping_runtime/harness/execute_frontier.py` | 接收当前 RGB、查询、候选前沿及其动作、原选项、已报告时间步，校验分数并把方向交给原版跟随器。后续可接自定义预测器。 |
| `mapping_runtime/harness/check_branch_io.py` | 验证同一初态执行不同候选动作后能采集到不同观测。 |
| `mapping_runtime/batch/README.md` | 冻结主基线与第二组的配置及权重，支持暂停、续跑、异常隔离和按同一回合配对计分。 |
| `mapping_runtime/external/scope/README.md` | SCOPE 原生 GOAT 和 MOST 多实例适配入口。 |
| `mapping_runtime/external/three_dmem/frozen_three_dmem.py` | 3D-Mem 冻结清单校验、单回合与批量执行入口，默认仅校验。 |
| `comparisons_nov14/mtu_native/README.md` | MTU3D 原生 GOAT、MOST 任务和两阶段模型配置。 |

原版 MOST-Nav 代码、场景、权重、逐帧图像和私有轨迹保存在服务器。冻结清单 SHA-256 为 `48f0f21020b3038121efd48e4f6dc880e47d3d1420c3a54745b3b55301d5cd3d`。
