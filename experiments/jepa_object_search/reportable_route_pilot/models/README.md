# 多实例寻物路线头

宿主导航系统先给出合法探索路线。这两个模型读取当前已经观察到的图像、目标文本和各路线的计划动作，预测短期路线能否首次进入某个未报告实例的有效观察范围。是否上报、是否命中目标及实际执行哪些动作仍由宿主决定。

| 版本 | 预测和打分 | 训练监督 |
|---|---|---|
| V7 | 修正 V5/V6 预测的未来局部特征，以目标文本、预测区域和之前实际看到的区域变化估计获得报告机会的概率。 | 同起点路线按新增可报告实例数排序，辅以是否有新增实例的二分类、真实未来局部特征余弦损失和候选间相对预测误差。 |
| V7S | 沿用 V7 学到的视觉修正作为初值。使用冻结的 16 个训练类别文本，在每条路线内部减去各词共有的得分，额外估计特定目标词与候选未来区域之间的关系。无词对照使用相同的位移先验。 | 在普通分叉和目标偏好相反的分叉上训练。若左侧路线更适合找椅子而右侧路线更适合找灯，则训练模型拉大这两个词对同一对路线的相对偏好。保留视觉预测损失。 |

## 输入

需要 Python 3、NumPy、PyTorch 和已有的 V5/V6 权重。视觉编码器使用冻结的 OpenCLIP RN50。此包直接读取本地预先编码的特征。所有输入向量均按训练时的相同图像裁剪和文本模板生成。自然冷启动样例没有既往报告记录，因此当前路线头没有已报告实例的视觉记忆输入。

- `FEATURES/words.npz` 以目标字符串为键，值为归一化文本向量 `[1024]`。提示为 `a photo of a <自然语言类别>.`，类别名称须先转为自然语言，如 `chair.n.01` 对应 `chair`。
- `FEATURES/views/<SHA256(sid)>.npz` 按起点存储。哈希输入是起点标识 `sid` 的 UTF-8 字节。`ctx[12,3,1024]` 记录每隔 30° 的真实观测，三个区域依次为整图、左侧和右侧居中高度的方形局部区域，边长为短边的 65%。`actions[K,24]` 为 K 条真实候选的计划动作，0、1、2 依次为前进、左转和右转。训练缓存还含 `teacher[K,5,3,1024]` 和 `samples=[2,4,6,12,24]`，这些是各候选执行后在相应步数看到的图像特征。验证只读取 `ctx` 和 `actions`。在线调用只传当前真实观察得到的视角。
- 官方、衍生任务各用一个目录，包含 `train.rows.json`、`calibration.rows.json`、`holdout.rows.json`。每行代表一个起点、一种目标和一个动作预算。`gain_newly_eligible` 记录各候选路线新进入报告范围的不同实例数，曾看见但尚未进入有效报告范围的实例仍然算数。`gain` 另行排除曾清晰看见的实例，作为辅助指标。

一行标签的格式如下。四步路线取自同一真实起点的候选计划。

```json
{"sid":"houseA/start0","house":"houseA","horizon":4,"step":0.25,"turn":0.5235987756,"query":"chair.n.01","action":[[0,0,0,0],[1,0,0,0]],"gain_newly_eligible":[1,0],"gain":[0,0],"reported_ids_before":[],"start_eligible_ids":[]}
```

## 运行

在脚本所在目录执行。V7 用 12 栋训练房屋和 4 栋校准房屋选择一次模型，独立评测使用另外 4 栋。V7S 沿用相同训练和校准房屋，最终检验使用第二组全新房屋。使用两个验证组时，相应官方与衍生任务目录里的 `holdout.rows.json` 必须配套。

```bash
python train_report_head.py fit --full-checkpoint v5.pt --sparse-checkpoint v6.pt --feature-dir FEATURES --official OFFICIAL --derived DERIVED --output RUN_V7
python train_report_head.py holdout --full-checkpoint v5.pt --sparse-checkpoint v6.pt --feature-dir FEATURES --official OFFICIAL --derived DERIVED --output RUN_V7
python train_semantic.py fit --full-checkpoint v5.pt --sparse-checkpoint v6.pt --feature-dir FEATURES --official OFFICIAL --derived DERIVED --output RUN_V7S --v7-run RUN_V7
python train_semantic.py check --full-checkpoint v5.pt --sparse-checkpoint v6.pt --feature-dir FEATURES --official NEW_OFFICIAL --derived NEW_DERIVED --output RUN_V7S --v7-run RUN_V7
```

推理时，`observed[V,3,1024]` 是实际已见的局部特征，`headings[V]` 是它们相对当前的弧度朝向，`actions[K,H]` 是候选动作前缀，`text[1,1024]` 是目标文本特征。下例计算稀疏历史下每条候选的分数，建议复用 `route_heads.choose_pair` 应用固定门槛。

```python
import torch
from route_heads import load_heads
from report_head import load_v7, forecast_inputs, route_scores
from semantic_head import restore, compose

with torch.inference_mode():
    frozen = load_heads("v5.pt", "v6.pt")
    future, nearest, motion = forecast_inputs(frozen, "sparse", observed, headings, actions)
    v7, alpha = load_v7("RUN_V7/v7.pt")
    v7_scores = route_scores(v7, future, nearest, motion, text, alpha)
    v7s, strength = restore("RUN_V7S/semantic.pt")
    semantic, _ = v7s(future, nearest, motion, text)
    v7s_scores, no_word_scores = compose(semantic, motion, strength)
```

训练种子、候选轮数、融合和准入门槛见 `v7_protocol.json` 与 `v7s_protocol.json`。分叉方向使用相同动作预算和传感器历史。统计保留所有可用起点及无机会回合，先合并同一起点的重复视角，再按房屋等权汇总。没有运行宿主 VLM 的 HSSD 对照让每个合法候选轮流充当参考方向。该指标衡量获得新报告机会的变化，整回合的实际成功率需要另外实测。`parity.json` 记录发布代码与实验代码在同一真实样本上的推理、实际训练损失及梯度的逐项核对结果。
