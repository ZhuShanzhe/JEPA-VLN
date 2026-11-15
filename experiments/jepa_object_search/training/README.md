# 首轮多实例寻物选路训练

[研究路线](../TRAINING_ROADMAP.md)给出首轮、跨类别泛化和回合内经验修正的研究问题。[数据准备](DATA.md)定义分支、收益与房屋划分。[模型与记忆接口](MODEL.md)说明两个选路头如何接收同样的观测和预测局部视觉特征。

| 数据来源 | 训练记录 | 开发记录 | 监督 |
|---|---:|---:|---|
| HSSD 自然分叉 | 144 | 48 | 实走过程中新增的持续实例证据 |
| HSSD 已报告实例对照 | 6 | 0 | 带报告画面和相对相机位置的分支排序 |
| Blender 单房颜色与报告对照 | 6 | 0 | 颜色查询互换、报告记忆互换 |
| Blender 隐藏布置 | 32 | 16 | 同一可见起点允许不同隐藏结果 |
| Blender 可见上下文 | 16 | 8 | 起点可见家具线索，目标实例在移动后才出现 |
| **合计** | **204** | **72** | **949 个候选条目，训练 698、开发 251** |

首轮使用冻结视觉与文本编码器，直接打分和 JEPA 两组分别训练 12 轮。离线指标按数据来源和标签分开统计。待两组权重能重新加载、指标与检查点摘要通过校验后，发布[结果目录](../pipeline/README.md)中的汇总。

## 数据与启动

运行根目录为服务器数据盘的 `/root/autodl-tmp/jepa_object_search`，首轮目录为其下的 `training_round1_20261115`。下列变量指向服务器现有环境。场景、逐帧观测、特征缓存和模型权重保存在数据盘。

```bash
ROOT=/root/autodl-tmp/jepa_object_search
R="$ROOT/training_round1_20261115"
KIT="$R/kit"
PY=/root/autodl-tmp/jepa_instance_bench_20261016/jev_vision_env/bin/python
```

1. `build_bays.py` 与 `build_room_cues.py` 由 Blender 导出房间和物体，`replay_bays.py` 用 Habitat 执行同起点分支。默认生成隐藏布置，`--cues` 生成两边均有可见上下文的组。 `mesa_software_device.c` 用于在指定的离屏采集进程中选择 Mesa 软件设备，渲染时可以腾出 GPU 给基线。
2. `data_prep.py`、`label_regions.py` 和 `verify_extension.py` 编制并核对 `pooled_final`。`local_features.py` 从原始 RGB 计算每条视觉序列的八个局部特征，生成 `grid_final`。`text_cache.py` 从本地 BGE-M3 编码目标文本，写入 `final_queries_bge.npz`。
3. `round_one.py freeze --round-dir "$R"` 锁定数据、代码、文本摘要和超参数，逐样本执行 CPU 前向检查。`category_splits.py` 生成后续七种类别留一的开发清单。首轮训练使用完整训练划分。
4. `launch_when_ready.py` 在数据盘记录等待计划，检查原版 300 回合的身份、结果文件、官方汇总、独立轨迹回放与诊断。验收通过且评测锁和显存可用时，启动 `round_one.py train`。直接访问训练入口也会复核基线。每个计划最多启动一次。

查看服务器等待状态及已验收的基线数量：

```bash
"$PY" "$KIT/launch_when_ready.py" --state-dir "$R/gate" check
cat "$R/gate/status.json"
```

同一计划的恢复命令如下。服务器实例保持运行时，等待程序可独立于 SSH 会话执行。如果实例重启，先恢复基线，再检查等待状态。同一状态目录会识别已运行或已完成的任务。

```bash
nohup /root/miniconda3/bin/python -u "$KIT/launch_when_ready.py" --state-dir "$R/gate" watch -- "$PY" -B "$KIT/round_one.py" train --round-dir "$R" >> "$R/watcher.log" 2>&1 < /dev/null &
```

训练输出位于 `first_fit/`，包括两个检查点和按房屋、任务类型汇总的 `receipt.json`。训练日志在 `gate/training.log`。发布器验证两个检查点后只向本项目仓库写入汇总 Markdown。
