"""Verify a completed two-arm run; optionally publish its aggregate Chinese report."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


OWNER = "git@github.com:ZhuShanzhe/JEPA-VLN.git"
REPO = Path("/root/autodl-tmp/lewm_research/github/LeWM_Navigation")
TOPIC = "jepa-image-nav-datasets-20261014"
DEST = "experiments/jepa_object_search/pipeline/frontier_head_round_01.md"
SOURCES = {"hssd":"HSSD", "blender":"Blender"}
LABELS = {"hssd_visible_evidence_delta":"可见证据增量", "controlled_prior_report":"已报告实例记忆对照",
          "controlled_color_query":"颜色查询对照", "new_sustained_exposure":"隐藏布置的持续曝光",
          "contextual_exposure":"上下文线索与目标曝光"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "Unexpected integer")
    return value


def number(value):
    require(type(value) in (float, int) and math.isfinite(value), "Invalid numeric result")
    return float(value)


def read_report(output):
    output = output.resolve(strict=True)
    require(all(p.is_file() and not p.is_symlink() for p in
                (output/"receipt.json", output/"direct.pt", output/"jepa.pt")), "Incomplete training outputs")
    receipt = json.loads((output/"receipt.json").read_bytes())
    require(receipt["format"] == "multi-source-frontier-v1" and
            set(receipt["arms"]) == {"direct", "jepa"}, "Expected two completed comparison arms")
    require(re.fullmatch("[a-f0-9]{64}", receipt["trainer_sha256"]) and
            re.fullmatch("[a-f0-9]{64}", receipt["dataset_sha256"]), "Missing input fingerprints")
    epochs = integer(receipt["epochs"], 1)
    seed = integer(receipt["seed"])
    data = receipt["split"]
    require(set(data) == {"train", "dev"}, "Unexpected splits")
    for split in data.values():
        states = integer(split["states"], 1)
        require(integer(split["houses"], 1) <= states and integer(split["branches"]) >= 2 * states,
                "Invalid decision-point counts")
        require(set(split["source_states"]) <= SOURCES.keys() and
                sum(integer(n) for n in split["source_states"].values()) == states, "Invalid domain counts")
        require(set(split["label_kinds"]) <= LABELS.keys() and
                sum(integer(n) for n in split["label_kinds"].values()) == states, "Invalid label counts")
    values = {}
    for mode, arm in receipt["arms"].items():
        require(sha(output / f"{mode}.pt") == arm["checkpoint_sha256"], "Checkpoint hash differs")
        require(len(arm["train_loss"]) == epochs, "Incomplete epoch history")
        for loss in arm["train_loss"]:
            number(loss)
        integer(arm["num_parameters"], 1)
        groups = {}
        by_domain, by_label = Counter(), Counter()
        for row in arm["dev"]:
            key = (row["source"], row["label_kind"])
            require(key[0] in SOURCES and key[1] in LABELS and key not in groups, "Unexpected development group")
            states = integer(row["states"], 1)
            houses = integer(row["houses"], 1)
            informative = integer(row["informative_states"])
            require(max(houses, informative) <= states, "Unexpected development group counts")
            accuracy = row["paired_accuracy_informative"]
            require((accuracy is None and informative == 0) or
                    (informative > 0 and 0 <= number(accuracy) <= 1), "Invalid ranking accuracy")
            require(number(row["mean_chosen_gain"]) >= 0 and number(row["mean_oracle_gap"]) >= -1e-6,
                    "Invalid gain")
            by_domain[key[0]] += states
            by_label[key[1]] += states
            groups[key] = row
        require(by_domain == data["dev"]["source_states"] and by_label == data["dev"]["label_kinds"],
                "Development coverage mismatch")
        values[mode] = groups
    direct, jepa = values["direct"], values["jepa"]
    require(direct.keys() == jepa.keys() and receipt["arms"]["direct"]["num_parameters"] ==
            receipt["arms"]["jepa"]["num_parameters"], "Arm coverage or capacity mismatch")
    lines = ["# 首轮多实例寻物选路头训练", "",
             "在按房屋隔离的开发集比较两种相同容量的选路头。两组使用相同的训练分支、查询和收益标签，"
             "JEPA 组另学习预测执行候选动作后的视觉特征。", "",
             f"随机种子 {seed}，每组训练 {epochs} 轮，每组可训练参数 {receipt['arms']['direct']['num_parameters']:,} 个。", "",
             "| 划分 | 房屋或受控布局数 | 决策点数 | 候选分支数 |", "|---|---:|---:|---:|"]
    for key, name in (("train", "训练"), ("dev", "开发")):
        v = data[key]
        lines.append(f"| {name} | {v['houses']} | {v['states']} | {v['branches']} |")
    lines += ["", "## 按数据来源与标签分组的离线选路结果", "",
              "| 来源 | 收益标签 | 方法 | 房屋或布局数 | 决策点数 | 有收益差异的决策点数 | "
              "平均选中收益 ↑ | 平均距最佳收益 ↓ | 有差异决策点的排序准确率 ↑ |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for source, label in sorted(direct):
        rows = (direct[(source,label)], jepa[(source,label)])
        for k in ("houses", "states", "informative_states"):
            require(rows[0][k] == rows[1][k], "Paired development denominators differ")
        for method, row in zip(("直接打分", "JEPA"), rows):
            acc = row["paired_accuracy_informative"]
            score = f"{acc:.1%}" if acc is not None else "—"
            lines.append(f"| {SOURCES[source]} | {LABELS[label]} | {method} | {row['houses']} | "
                         f"{row['states']} | {row['informative_states']} | {row['mean_chosen_gain']:.4f} | "
                         f"{max(0.,row['mean_oracle_gap']):.4f} | {score} |")
    lines += ["", "选中收益与距最佳收益按本组决策点平均。排序准确率先在每个有真实收益差异的决策点内"
              "比较候选对，再对这些决策点求平均。各标签采用各自的收益定义。这些数值度量离线分支选择，"
              "完整导航回合的配额完成率和实例召回需闭环测试。"]
    if any(label == "new_sustained_exposure" for _, label in direct):
        lines += ["", "隐藏布置对照包含此前观测相同、遮挡后目标位置不同的场景。线索出现前，多个真实未来"
                  "与同一输入相容。这里的排序数字只描述这批分支的实测表现，单次未选中高收益方向无法证明模型辨识失败。"]
    if all("mean_chosen_minus_uniform" in row for groups in values.values() for row in groups.values()):
        lines += ["", "## 随机选路参照与区域概率", "",
                  "| 来源及标签 | 方法 | 相对均匀选路的平均增益 | 区域正例 / 负例 | Brier 分数 ↓ |",
                  "|---|---|---:|---:|---:|"]
        for key in sorted(direct):
            for method, row in (("直接打分",direct[key]),("JEPA",jepa[key])):
                n_pos,n_neg=integer(row["region_n_pos"]),integer(row["region_n_neg"])
                brier=row["observed_region_brier"]
                require((n_pos+n_neg==0 and brier is None) or
                        (n_pos+n_neg>0 and brier is not None and 0<=number(brier)<=1),"Invalid regional probability metrics")
                value="—" if brier is None else f"{brier:.4f}"
                gain=number(row["mean_chosen_minus_uniform"])
                lines.append(f"| {SOURCES[key[0]]} · {LABELS[key[1]]} | {method} | {gain:.4f} | {n_pos} / {n_neg} | {value} |")
        lines += ["", "Brier 分数只统计区域存在标签已核验的候选。仅有负例的组无法估计对正例的召回。"]
    lines += ["", f"训练数据指纹 `{receipt['dataset_sha256']}`，训练器指纹 `{receipt['trainer_sha256']}`。"]
    return "\n".join(lines) + "\n", receipt


def publish(body):
    repo = REPO.resolve(strict=True)
    def git(*args, data=None, env=None, check=True):
        return subprocess.run(["git", "-C", str(repo), *args], input=data, stdout=subprocess.PIPE,
                              text=True, check=check, env=env, timeout=180)
    def val(*args, **kwargs):
        return git(*args, **kwargs).stdout.strip()
    def ancestor(old, new):
        return git("merge-base", "--is-ancestor", old, new, check=False).returncode == 0
    def checkout_clean():
        require(val("rev-parse", "--show-toplevel") == str(repo) and val("branch", "--show-current") == TOPIC
                and not val("status", "--porcelain", "--untracked-files=normal"), "Research checkout not clean or wrong branch")
        for mode in ([], ["--push"]):
            require(val("remote", "get-url", *mode, "--all", "origin") == OWNER, "Unexpected Git remote")
    checkout_clean()
    gitdir = Path(val("rev-parse", "--absolute-git-dir"))
    with (gitdir / "frontier-training-publication.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkout_clean()
        git("fetch", "--no-tags", "origin", "+refs/heads/main:refs/remotes/origin/main",
            f"+refs/heads/{TOPIC}:refs/remotes/origin/{TOPIC}")
        main, topic = (val("rev-parse", "refs/remotes/origin/" + ref) for ref in ("main", TOPIC))
        base = topic if ancestor(main, topic) else main
        require(ancestor(main, base) and ancestor(topic, base) and ancestor(val("rev-parse", "HEAD"), base),
                "Branches diverged or checkout contains unpushed work")
        present = git("cat-file", "-e", f"{base}:{DEST}", check=False)
        if present.returncode == 0:
            require(git("show", f"{base}:{DEST}").stdout == body, "Existing first-round report differs")
            target = base
        else:
            # Isolated index: a concurrent staging operation cannot enter this publication.
            with tempfile.TemporaryDirectory() as tmp:
                env = {**os.environ, "GIT_INDEX_FILE":str(Path(tmp)/"index")}
                git("read-tree", base, env=env)
                blob = val("hash-object", "-w", "--stdin", data=body)
                git("update-index", "--add", "--cacheinfo", "100644", blob, DEST, env=env)
                require(val("diff", "--cached", "--name-only", base, env=env) == DEST, "Unexpected publish set")
                git("diff", "--cached", "--check", base, env=env)
                tree = val("write-tree", env=env)
                target = val("commit-tree", tree, "-p", base, "-m", "Report first-round JEPA frontier training")
        checkout_clean()
        git("push", "--atomic", "origin", f"{target}:refs/heads/main", f"{target}:refs/heads/{TOPIC}")
        remote = {ref:commit for commit,ref in (line.split() for line in
                  val("ls-remote", "--refs", "origin", "refs/heads/main", f"refs/heads/{TOPIC}").splitlines())}
        require(remote == {"refs/heads/main":target, f"refs/heads/{TOPIC}":target}
                and git("show", f"{target}:{DEST}").stdout == body, "Remote verification failed")
        checkout_clean()
        git("merge", "--ff-only", target)
    return target


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path, help="completed directory created atomically by trainer.py")
    ap.add_argument("--publish", action="store_true", help="push the single verified aggregate report")
    args = ap.parse_args()
    body, receipt = read_report(args.output)
    if args.publish:
        require(sha(Path(__file__).with_name("trainer.py")) == receipt["trainer_sha256"], "Trainer version differs")
        print(publish(body))
    else:
        print(body, end="")
