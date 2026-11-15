# HSSD 分支采集与验收

`import_official_hssd.py` 导入官方 ObjectNav 任务包。`wide_queue.py` 根据固定的房屋划分组织采集，`collect_branches.py` 在 Habitat 中实际执行各分支。`offline_branch_loader.py` 分开读取在线输入和未来监督，`cache_features.py` 提取冻结 V-JEPA 2.1 特征，`wide_audit.py` 核对划分、帧数、动作与特征尺寸。脚本沿用实验服务器的 Habitat 环境、数据根目录和 VLFM 前沿提取依赖。
