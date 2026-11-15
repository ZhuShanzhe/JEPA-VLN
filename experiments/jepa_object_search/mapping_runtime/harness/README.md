# 建图与动作执行

`run_pipeline.sh` 启动建图服务或调用已配置的 MOST-Bench 评测器。`prepare_tasks.py` 准备任务输入，`start_mapping.py` 加载建图服务。`execute_frontier.py` 只在宿主提出可选方向时接入评分器，再交回原路径跟随器执行。

`map_probe.py` 核验真实位姿和地图点，`check_branch_io.py` 核验相同初始观测能沿两条路线采集到不同的未来图像。`verify_branch.py` 与 `phase_receipt.py` 核验控制器执行记录。[操作命令与验收结果](../../pipeline/README.md)
