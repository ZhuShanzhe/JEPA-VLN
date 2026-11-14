# 方向预测与报告记忆

从项目根目录运行 `python frontier_jepa/run_pilot.py --output frontier_jepa/runs/hssd_v2_seed17_e5 --seed 17 --epochs 5`，再运行 `python ops/check_frontier_routing.py`。

`ablate_readout.py --output DIR` 运行三个种子的监督和读出消融。`vision.py` 使用冻结的 V-JEPA 2.1；`memory.py`、`fit_report_fixture.py` 和 `score_reports.py` 用于独立的单房报告实验。HSSD 接口采用局部前/左坐标、24 个原子动作、每转 30°。Blender 示例采用相对 X/Z、19 个动作、每转 15°，使用独立权重。
