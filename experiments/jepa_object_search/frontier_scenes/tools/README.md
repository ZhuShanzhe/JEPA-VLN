# 场景生成与回放

用 Blender 3.6.23 运行 `blender -b --python tools/build_room.py -- room`，在 Habitat-Sim 0.3.3 下运行 `python tools/replay_room.py` 生成等预算分支。

`most_fixture_mesh.py`、`most_fixture_inputs.py` 和 `most_fixture_policy.py` 为原版 MOST 评测器构造两个自建房间测试回合，`verify_most_cli.py` 检查行为与得分。
