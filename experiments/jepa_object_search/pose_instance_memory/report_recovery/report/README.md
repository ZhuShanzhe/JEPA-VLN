# 报告前的现场确认与动作恢复

[报告控制器](report_recovery.py)在主动报告和自动报告两个入口共享现场核验。当证据足够时，沿用原生报告动作与回执登记。证据不足时，将拒绝原因反馈给上层，并选择接近、重新取证或探索动作。[当前画面核验](report_evidence.py)利用当前 RGB 检测框和登记的参考视图核对实例，区分同类候选。

[动作与证据测试](test_report_recovery.py)通过上一阶段相同的原生接口检查报告与恢复。整机入口为 [RepairFrontier](../repair_agent.py)，在[上一版闭环策略](../../closed_loop_pilot/trace_agent.py)上同时加载报告恢复与[延迟身份复核](../pose/delayed_pose.py)。在配置好 MOST-Nav、Habitat 和项目既有模型依赖的评测环境中，把这两个模块目录与原有基线模块目录加入 Python 搜索路径，运行 `python -m unittest test_report_recovery test_delay`。
