# MOST-Nav 基线评测

[建图与动作执行](harness/README.md)连接 Habitat、VGGT-SLAM、作者的 NavAgent 和 MOST-Bench 裁判。[观测器](probes/README.md)记录候选方向。[批量评测](batch/README.md)固定 300 个 ManyON 回合及其配置。[离线诊断](analysis/README.md)区分路线覆盖、实际曝光和报告质量。

代码对应服务器的数据盘目录布局。运行依赖单独获得的作者代码、许可场景、模型权重、评测环境、私有视觉模型服务设置及冻结清单。`weights.env` 列出本地建图权重的位置。数据和软件版本参见[流程验收](../pipeline/README.md)。
