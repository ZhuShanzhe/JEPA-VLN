# 候选框与 MOST-Nav 的接口

导航代理入口为 `provider_agent:BoxComparisonAgent`。在服务器的研究项目目录中加载 `source jepa_box_v26/adapter/environment.sh`，设定 `NAV_LOCAL_BOXES=1`，再按下表选择 `NAV_BOX_PROVIDER`。图像输入为 RGB。检测输出使用原图坐标，顺序为左、上、右、下，即 `[x_min, y_min, x_max, y_max]`。登记、根据当前图确认实例、报告和环境裁判沿用原有接口。

| 模式 | 上层收到什么 | 当前帧如何核对 |
|---|---|---|
| `dino`（默认） | 原有掩码菜单与当前自动候选 | 直接使用原 Grounding DINO 工作者 |
| `native` | 原建图服务为该帧返回的 SAM 区域及编号 | 本地 SAM 提框，冻结 CLIP 按类别打分，取正分的前三个供实例匹配 |
| `refined` | 原接口读取训练精修后的 DINO 候选 | 相同检测器和名额，对坐标调整后进入原匹配器 |

原生菜单中的区域编号由建图服务生成，局部比较产生的掩码编号只在对应图像中使用。接口验收使用三张重新渲染的历史图像，逐项核对代理输出的坐标和缓存，并用固定的建图菜单验证原编号能通过上层选择入口。32 项既有的验证、复看、证据交接和候选测试通过。

时钟复看分别使用先前登记的框、同一方法在登记图重选的首框构建外观参考。每种方法均在登记图通过自匹配，在实际靠近后的图像中仍有可用候选，但均因 `no_current_instance_match` 暂缓报告。该试验把问题定位到跨视角确认，不能区分尺度变化、参考图细节与匹配器阈值各自的作用。

[代理适配](provider_agent.py) · [本地提框服务](box_service.py) · [接口验收](check_interfaces.py) · [接口结果](interface_results.json) · [登记图与新视角对照](source_to_live.py) · [对照结果](source_to_live.json)
