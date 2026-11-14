# 选路接口

在宿主解析出具体动作后调用 `choose_frontier`，并将返回的前沿编号交给宿主验证。默认旁路评分，只有在 `apply=True` 且收到合法的探索决策时才替换方向。

从项目根目录运行 `python ops/check_frontier_routing.py` 读取 HSSD 开发缓存和种子 17 的预测权重。
