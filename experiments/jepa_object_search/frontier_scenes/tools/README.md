# 场景生成与回放

用 Blender 3.6.23 运行 `blender -b --python tools/build_room.py -- room`，在 Habitat-Sim 0.3.3 下运行 `python tools/replay_room.py` 生成等预算分支。

`most_fixture_mesh.py`、`most_fixture_inputs.py` 和 `most_fixture_policy.py` 为原版 MOST 评测器构造两个自建房间测试回合，`verify_most_cli.py` 检查行为与得分。

`audit_assets.py` 从作者任务包生成 300 回合冻结清单，并核对指标和任务统计。将授权版本置于项目运行目录的 `frontier_scenes/most_release/`，保留原始下载包 `frontier_scenes/downloads/most_release.zip`，并准备 HSSD ObjectNav v0.2.5 官方训练与验证任务。从项目运行根目录执行 `python frontier_scenes/tools/audit_assets.py`。产出的清单与校验记录供 `mapping_runtime/harness/prepare_tasks.py` 使用。
