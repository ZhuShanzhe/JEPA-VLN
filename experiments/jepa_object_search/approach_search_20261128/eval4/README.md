# 独立房屋检验

[回合清单](panel.json)记录独立房屋、起点编号、类别、配额和任务分片哈希。每个房屋选择一个类别；[起点连通性](eligibility.json)在策略行动前检查至少有两个实例的观察位从起点可达。与[同房校正](../eval3/README.md)使用相同冻结跟随器、学习先验和视觉分数。

在私有运行目录中执行 `python approach_search_20261128/prepare_replay.py . eval4`。所有第三方依赖与本地训练文件均按[本批哈希](locked_sources.json)校验。随后从 Habitat 环境运行 `python approach_search_20261128/eval4/physical_panel.py --indices 0 1 2 3 4 5 6 7 --arms old new prior_only`，再运行 `python approach_search_20261128/eval4/accept.py`。结果分别记录不同实例数、达到双实例配额的回合数、选定观察位的到达次数和碰撞次数。
