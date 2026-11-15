"""Render sanitized native ManyON results after the official queue and offline audit."""
import json
import math
from pathlib import Path


CONDITIONS = {"all": "全部", "many_category": "类别寻物", "many_attribute": "属性寻物"}
PANEL = "48f0f21020b3038121efd48e4f6dc880e47d3d1420c3a54745b3b55301d5cd3d"


def read(path):
    return json.loads(path.read_text())


def percentage(value):
    if value is None:
        return "—"
    if not (math.isfinite(value) and 0 <= value <= 1):
        raise ValueError("Expected a unit-interval metric")
    return f"{100 * value:.1f}%"


def count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Expected a nonnegative count")
    return value


def render(audit, official, identity=None, visibility=None):
    if not (audit["complete"] and audit["accepted"] == official["accepted"] == 300
            and official["panel_sha256"] == PANEL):
        raise ValueError("Full frozen baseline required")
    expected = {"all": 300, "many_category": 150, "many_attribute": 150}
    lines = ["# MOST-Nav 多实例寻物基线", "",
             "使用冻结的 36 栋 HM3D 房屋和 300 个文本 ManyON 回合。类别和属性任务各 150 个，"
             "每回合要求找到 2 至 10 个不同目标，最多执行 300 个动作。主策略为作者的 NavAgent，"
             "建图使用 VGGT-SLAM 与 GPS 尺度，图像描述和决策模型固定为 `gpt-4o-2024-11-20`。", "",
             "## 正式性能", "",
             "| 条件 | 回合数 | 完成配额 SR | 路线接近配额 OSR | 实例召回 | 报告精确率 | F1 | SPL-Multi |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, label in CONDITIONS.items():
        g = audit["groups"][key]
        n = count(g["n"])
        if n != expected[key]:
            raise ValueError("Incomplete condition")
        metric = official["metrics"][key]
        cells = [percentage(metric[k]) for k in ("sr", "osr", "recall", "precision", "f1", "spl_multi")]
        lines.append(f"| {label} | {n} | " + " | ".join(cells) + " |")
    lines.extend(["", "OSR 用实际走过的路线计算。机器人只要接近了足够多不同目标的合法观察位置，就满足路线条件。"
                  "SR 还要求它在允许的范围内实际发出有效报告。", "",
                  "## 失败发生在哪一步", "",
                  "| 条件 | 完成配额 | 实际路径接近的合格目标少于配额 | 已接近足够多、有效报告不足 | 无法归类 |",
                  "|---|---:|---:|---:|---:|"])
    for key, label in CONDITIONS.items():
        f = audit["groups"][key]["failure_partition"]
        cells = [count(f[k]) for k in ("success", "insufficient_approach_quota",
                                      "sufficient_approach_but_underreported", "diagnostic_unknown")]
        if sum(cells) != expected[key]:
            raise ValueError("Failure partition does not close")
        lines.append(f"| {label} | " + " | ".join(map(str, cells)) + " |")
    g = audit["groups"]["all"]
    f, gap = g["failure_partition"], g["quota_shortfall"]
    exploration = count(f["insufficient_approach_quota"])
    report = count(f["sufficient_approach_but_underreported"])
    missing = count(gap.get("missing_to_reach_quota", 0))
    missed_reports = count(gap.get("reached_but_unreported", 0))
    lines += ["", f"按缺少的实例数统计，实际路线距离配额还差 {missing} 个实例。"
              f"另外 {missed_reports} 个配额缺口出现在已经接近、但有效报告未跟上的环节。"
              f"另有 {count(gap.get('unresolved', 0))} 个缺口缺少足够几何证据。一个回合可能同时存在两种缺口。"
              "观察点可以重叠，一次报告也可能匹配多个实例。", "",
              "## 对方向预测研究的意义", ""]
    front = g["frontiers"]
    at = lambda k: count(front.get(k, 0))
    lines += [f"原版策略经过校验的探索前沿请求共 {at('requested')} 次，其中 {at('blocked')} 次执行受阻。"
              f"有 **{at('applied_with_alternative')} 次**在至少两个合法候选中做了选择并激活路线，"
              "这些是现有方向回调可以直接干预的决策。"
              f"其中 {at('windows_with_geometry')} 个决策窗口可重建几何结果，"
              f"{at('windows_with_new_approach')} 个在下一次决策前首次接近新目标。短窗口内未接近目标也可能是在穿越走廊。", ""]
    if identity is not None:
        lines += [f"有 {count(identity['early_incomplete_and_repeat'])} 个回合同时出现重复报告和未完成配额便主动结束。"
                  f"其中可直接从逐次报告追溯到“不同策略实例编号、同一裁判目标”的回合数为 "
                  f"{count(identity['early_repeat_with_distinct_local_ids'])}。未能对齐策略编号的报告没有计入后者。", ""]
    if visibility:
        lines += [visibility, ""]
    reps = g["report_events"]
    lines += [f"裁判收到了 {count(reps.get('actions',0))} 次报告动作。"
              f"{count(reps.get('no_eligible_target_in_range',0))} 次上报时附近没有合格目标，"
              f"{count(reps.get('with_over_quota_claim',0))} 次包含超额或重复认领。"
              "这些情况应分别检查定位、识别与实例去重。", ""]
    ci = g.get("ci95")
    if ci:
        lo, hi = ci["sr"]
        lines += [f"全体回合 SR 的场景级 95% bootstrap 区间为 {percentage(lo)}–{percentage(hi)}"
                  f"（{count(g['scene_count'])} 栋房屋，2,000 次重采样）。", ""]
    if exploration or report:
        lines += [f"有 **{exploration} 个回合**按原路径无法覆盖目标配额。这一组是评估前沿选择模块的切入点。"
                  "需要进一步区分 VLM 有多个可达前沿时选错方向、建图没有提供合适候选，以及低层跟随未能执行选择。"
                  "有多个候选且实际执行的方向长期没有带来新实例，是最值得用独立训练房屋构造分支样本的情形。", "",
                  f"另有 **{report} 个回合**原路径已经接近足够多的目标，但最终报告不足。"
                  "这部分应同时考察实例确认、报告时机和去重。仅凭靠近合格观察点无法断定摄像机看清了目标，"
                  "也无法断定换一个方向就能提高成功率。", ""]
    bottleneck = g.get("frontiers_by_failure_partition", {}).get("insufficient_approach_quota", {})
    alternatives = count(bottleneck.get("episodes_with_prequota_alternative", 0))
    windows = count(bottleneck.get("windows_before_approach_quota", 0))
    no_gain = count(bottleneck.get("prequota_windows_without_new_approach", 0))
    lines += [f"在路线覆盖不足的 {exploration} 个回合中，有 {alternatives} 个实际出现过模块可改选方向的机会。"
              f"这些回合有 {windows} 个可核对的多候选探索窗口，其中 {no_gain} 个在下一次决策前没有新增可报告目标。"
              "这是方向预测模块可干预的观测范围。", ""]
    lines += ["失败类别来自已执行轨迹与裁判的事后核对。未执行方向的真实收益需要在独立开发场景中，"
              "从同一状态执行多个候选方向并比较。冻结评测的图像、动作和目标标注不进入训练。", ""]
    return "\n".join(lines)
