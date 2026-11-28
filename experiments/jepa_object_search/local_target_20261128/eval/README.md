# 仿真与评分

[场景清点](inventory.py)、[回合筛选](freeze.py) 和[建筑分组筛选](family_freeze.py) 生成最终的八栋建筑清单。[策略冻结](seal_local.py) 记录训练权重、本地推理实现、旧策略依赖和清单的 SHA-256。两个测试臂共用旧版的仿真器、传感器、运动模型及评分器。[独立验收](../eval_family/README.md)

在私有科研运行目录执行如下命令。准备工作依赖已配置的 Habitat 环境、StopWatch HSSD-OV 数据、上一版冻结模型和本次表面读出权重。旧版的本地依赖可通过 [prepare_replay.py](../../approach_search_20261128/prepare_replay.py) 核对。

```sh
PY=simulation/habitat-env/bin/python
MOD=local_target_20261128
$PY "$MOD/eval/freeze.py"
$PY "$MOD/eval/family_freeze.py"
$PY "$MOD/eval/seal_local.py" "$MOD/eval_family"
$PY "$MOD/eval_family/physical_panel.py" --indices 0 1 2 3 4 5 6 7 --arms new
$PY "$MOD/eval/run_local.py" --panel-dir "$MOD/eval_family" --indices 0 1 2 3 4 5 6 7
$PY "$MOD/eval/diagnose.py" "$MOD/eval_family"
$PY "$MOD/eval_family/paired_audit.py" "$MOD/eval_family"
```

[开发屋短回合](smoke.py) 用于检查仿真和模型连接。[逐阶段诊断](diagnose.py) 从私有日志核实目标编号、位置、朝向、释放原因与首次靠近时执行的动作阶段。建筑评测的公开文件只保留汇总。
