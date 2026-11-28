# 同房校正

[回合清单](panel.json)与第二批开发起点相同。三组使用冻结的观察位跟随器和同一套候选生成器。旧方法按短期目标视觉预测排序，纯先验组按摆放先验排序，新方法使用摆放先验、地图记忆和原视觉证据。视觉证据直接使用 V16 的每六步概率。

在配有 Habitat、HSSD-OV 任务与场景、MOST-Nav 源码和项目模型的私有运行目录中，运行 `python approach_search_20261128/prepare_replay.py . eval3`。程序根据[文件哈希](locked_sources.json)检查既有模型并恢复本批冻结模块。历史 HSSD 跟随器的哈希以 `a975a4b8` 开头。[归档执行器](../../route_execution_v1/point_executor_hssd_frozen.py)

在 Habitat 环境中执行 `python approach_search_20261128/eval3/physical_panel.py --indices 0 1 2 3 4 5 6 7 --arms old new prior_only`，再执行 `python approach_search_20261128/eval3/accept.py`。传感输入包含仿真提供的位置与朝向。独立评分按每个实例的合法观察位统计 300 步内接近的不同实例数。
