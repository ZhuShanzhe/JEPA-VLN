# 随探索更新路线选择

控制器在同一栋房屋连续走四段。初始化时传入目标词与训练种子，每次转头取得三张当前 RGB，传给 `choose_after_scan` 连同三条各 24 步的候选动作计划。它返回选中的序号、这条路线最可能看到的环境，以及本轮探索策略是否改变了原首选。走完路线后，`consume_walk` 接收实际执行动作、逐步碰壁标记和第 6、12、18、24 步的四帧图像。下一次选路才使用这些反馈。换回合用 `reset(target)` 清空本屋记录。[可嵌入控制器](live_search.py) · [空间与环境记忆](scene_memory.py)

| 文件 | 用途 |
|---|---|
| [图像线索校准](calibrate_cues.py) · [校准结果](cue_calibration.json) | 在训练房学习 RGB 目标线索与首次出现的关系，把偏弱或重复证据的影响限制在较小范围。 |
| [编码采集轨迹](encode_rollouts.py) · [连续选路](run_search.py) | 同一物理路口的编码可供多个方法共用，每次决策只读取已实际选择的前缀。校准房决定记忆和探索权重。 |
| [比较选路方法](analyze.py) · [校准配置](online_selection.json) | 同一起点、目标、模型种子、候选路线和步数配对，以房屋为单位计算差异区间。 |
| [环境、先验与探索](fresh_summary.json) · [长程选路](fresh_horizon_summary.json) · [本屋布局预报修正](fresh_layout_summary.json) | 原有选路头、环境 JEPA、同输入直接监督、在线记忆和长程价值在留出房的独立结果。 |
| [可达机会分析](diagnosis.py) · [汇总](opportunity.json) | 穷举可行路线所得的事后机会只用于评估，以判断还有多少改进空间。 |
| [动作及首次观察阶段](route_behavior.py) · [汇总](route_behavior.json) | 按相同起点和种子更换目标词，统计首选是否改变，并核查关闭特定记忆项后的轨迹与实例差异。 |
| [逐次实拍 RGB 验证](check_controller.py) · [结果](controller_check.json) | 从同一起点开始完全依靠原始观测选路，逐个路口核对缓存评测的选择。 |

默认使用已训练的长程价值、当前场景的访问记忆和目标证据。可设置 `layout=True` 启用根据本屋经验修正房型预报的附加模块。部署目录使用 `jepa_env_v13/layout_update` 存放该模块与冻结系数，仓库中对应目录为 `layout`。训练权重、原始观测和逐回合轨迹保存在实验服务器数据盘。
