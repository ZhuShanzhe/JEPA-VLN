# 选路头训练、局部视觉特征和回合内经验

`trainer.py` 从每个决策状态读取当前视觉特征、候选方向、原子动作序列、实际步长和转角、目标查询，以及可获得的历史报告特征和相对位置。未来特征、实例增益及可用的区域标签只进入损失或离线评测。视觉投影只用训练房屋的历史帧拟合。预测器用 GRU 编码计划动作，再预测沿路线取得的视觉特征；读出头按查询对局部 token 加权并结合报告记忆打分。

## 特征与训练接口

以下命令从已配置的服务器项目运行根目录执行。`P` 为含 PyTorch、timm、PIL 和 transformers 的 policy 环境 Python。执行这些命令时，数据、权重和逐状态特征保存在运行目录。

```bash
R=training_round1_20261115
P=/root/autodl-tmp/jepa_instance_bench_20261016/policy_env/bin/python
"$P" "$R/kit/local_features.py" --root "$PWD" --data "$R/pooled_final" \
    --extra-raw "$R/bay_raw" --extra-raw "$R/bay_cues" \
    --reuse-grid "$R/grid_mixed" --reuse-pooled "$R/pooled_mixed" --output "$R/grid_final" --device cpu
"$P" "$R/kit/text_cache.py" --data "$R/grid_final" --model mapping_runtime/weights/bge-m3 \
    --output "$R/final_queries_bge.npz" --device cpu
"$P" "$R/kit/trainer.py" --data "$R/grid_final" --text-cache "$R/final_queries_bge.npz" --device cpu --check-only
```

局部视觉缓存用 V-JEPA 2.1 ViT-B/16 的 4 帧输入，保留两个时间位置，每个位置按四个空间象限取平均，共 8 个 768 维局部特征。重编码逐条核对八个局部特征的均值与原始池化特征。复用已有缓存时，额外核对原输入和输出的校验值、索引、候选顺序、标签和全部非视觉字段。文本编码使用本地 BGE-M3 的归一化 CLS 表示；运行时用 `text_cache.GoalEncoder(model_dir).encode([query])` 处理相同规范的实际文本。未提供文本缓存时训练器使用分词哈希嵌入，两种接口须与自身检查点匹配。

训练进程以相同初始化、样本顺序、参数量和更新次数训练 direct 与 jepa。两组均使用同状态非平局分支的排序损失和真实可用的区域存在监督，未知区域标签为 -1。JEPA 组增加冻结未来特征的余弦预测损失。训练输出目录须为新路径；两组共同完成后原子发布 `direct.pt`、`jepa.pt` 和 `receipt.json`。准入验证之后，用上述训练器命令替换 `--check-only` 为 `--output 新路径 --seed 17 --epochs 12`。检查点可通过 `trainer.load_checkpoint(path)` 恢复，`RouteHead.forward` 的参数仅含在线字段。

开发成绩按场景来源和标签类型单列。mean_chosen_gain 为所选分支在该数据标签下的平均增益，mean_oracle_gap 为同起点最优分支相对所选分支的差值，paired_accuracy_informative 只统计真实收益有差异的状态。收益不同与选择前是否能观察到区分线索分别衡量。遮挡后的真实布置可以改变收益，但决策前输入完全相同的样本应给出相同预测。新类别上的收益和闭环配额成功率需要对应的独立评测。

mean_uniform_gain 是同起点均匀选路的期望增益，mean_chosen_minus_uniform 是实际选择与此参照的平均差。observed_region_brier 只在拥有可靠区域存在标注的候选上按候选数计算，连同 region_n_pos 和 region_n_neg 报告；没有标签时该值为空。只有单侧标签的组不足以评定对两类的区分能力。

## 场景内经验修正

`episode_evidence.py` 实现一个可旁路的回合内记忆。传入的区域 ID、入口线索向量、实际抵达状态、检测和覆盖置信度均来自在线系统。每次分叉先用新观测在地图中提取**各入口各自的视觉描述**，再用已执行路线的可靠证据调整相似、尚未访问的区域。目标查询或回合变化时隔离支持记录。同一区域低置信的重复观察不会增加证据量。代码只对实际执行的选择接受反馈。部署需要宿主在跨决策时间内维护区域关联、获取可核验的检测与覆盖估计；独立的分支样本只能诊断和训练选路头。

```python
from episode_evidence import Entrance, EpisodeEvidence

memory = EpisodeEvidence(strength=0.35, minimum_coverage=0.8, score_unit=1.0)
memory.reset(episode_id)
entrances = [Entrance(region, vector) for region, vector in visible_entrance_descriptors]
scores = memory.scores(episode_id, query_text, entrances, original_scores)
# The host applies the route and identifies the executed choice.
memory.begin_choice(episode_id, query_text, entrances[executed_index])
memory.finish_choice(episode_id, arrived=arrived, target_seen=visually_confirmed,
                     coverage=observed_coverage, confidence=observation_confidence)
```

区域概率尚未校准时，省略 `prior_presence`，权重残差相对于中性先验计算。收益排序分数不能直接解释为存在概率。`score_unit` 是固定分数单位，可在开发房屋选定。回合独立，开始新回合时调用 `reset`。`python episode_evidence.py` 用规律相反和无共享规律的合成对照检查重置、查询隔离、不充分观察及未抵达的处理；这些数字仅描述接口行为。
