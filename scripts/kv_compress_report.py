"""Generate the KV cache compression survey and recommendation Word report.

Reads the two survey results bundled below (quantization/latent + eviction/merge/architecture),
adds the expert-practice analysis and the prioritized plan, and writes one .docx to
~/fusion/architecture/. Run inside the fusion venv: python scripts/kv_compress_report.py
"""

import logging
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

logger = logging.getLogger(__name__)

OUT = Path.home() / "fusion/architecture/fusion-kv-cache-compression-plan-1105.docx"

PLACEHOLDER = "PENDING"


def style(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "PingFang SC"
    normal.font.size = Pt(11)


def h(doc: Document, text: str, level: int) -> None:
    doc.add_heading(text, level=level)


def p(doc: Document, text: str) -> None:
    doc.add_paragraph(text)


def bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, head in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = head
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = str(value)
    doc.add_paragraph()


def build() -> Path:
    doc = Document()
    style(doc)

    title = doc.add_heading("fusion 本地部署 KV Cache 压缩：论文综述与优先级方案", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p(doc, "日期：2026-11-05  |  范围：不牺牲精度的 KV cache 压缩（RULER/LongBench 掉点 < 1%）")
    p(doc, "硬件前提：Apple Silicon M5 Max，128GB 统一内存；推理引擎 fusion-mlx（MLX/Metal）；"
           "已有资产：LSA/DSA 稀疏选择（Ward 聚类 + top-p bucket）、TurboQuant 式 KV 量化、paged SSD KV cache。")

    h(doc, "1. 问题定义与判据", 1)
    bullets(doc, [
        "压力来源：统一内存 = 权重 + KV cache + 系统共用；27B bf16 权重约 52GB，长上下文 KV cache 随 token 数线性增长，"
        "GQA 下 27B（4 KV 头 × 256 维 × 64 层）每 token 全量 KV 约 256KB bf16，1M token 即 256GB——全量存放不可行。",
        "压缩判据（不牺牲精度）：RULER、LongBench、∞-bench、needle 多深度多密度；ppl 漂移 < 0.1；"
        "长程检索召回与 dense baseline 差距 < 1%。",
        "压缩对象三个正交维度：每个 token 的 KV 字节数（量化/低秩）、保留多少 token（选择/合并/淘汰）、"
        "KV 放在哪里（统一内存 / SSD 分层）。",
    ])

    h(doc, "2. 方法族全景", 1)
    table(doc,
          ["方法族", "代表工作", "典型压缩比", "精度风险", "与 fusion 现有资产关系"],
          [
              ["KV 量化（K/V 位宽）", "KIVI, KVQuant, SKVQ, TurboQuant", "4x-8x", "中（需离群处理）", "已有 TQ 基础，直接演进"],
              ["低秩/latent KV", "MLA, CLA, YOCO, 层间共享", "10x-90x", "架构级改造，需训练", "GLM-5.3-Flash latent-KV 已是此路线"],
              ["token 选择/稀疏", "H2O, SnapKV, Quest, LSA/DSA", "4x-16x（读取侧）", "低（KV 全存，读取稀疏）", "fusion-lsa 已移植 P0"],
              ["token 合并", "CaM, KVMerger, MiniCache", "2x-5x", "中", "待评估"],
              ["淘汰/流式", "StreamingLLM, TOVA", "无限上下文", "中高（丢信息）", "paged SSD 可承接溢出"],
              ["分层存储", "paged SSD KV, offload", "容量换位宽", "低（延迟换）", "已有 paged-SSD-KV"],
          ])

    h(doc, "3. 论文逐一分析", 1)
    p(doc, "标注说明：arxiv id 未加注为高置信；(~) 表示搜索环境受限下未能在线核验（MiKV、TurboQuant、1-bit KV 三篇）。"
           "选择/合并族的 18 篇 id 已通过 arxiv API 在线核验。")
    h(doc, "3.1 量化与 latent/低秩族（15 篇）", 2)
    table(doc,
          ["论文", "方法一句话", "压缩比", "精度影响", "MLX 适配 0-3"],
          [
              ["KIVI (2402.02750, 2024)", "非对称 2bit：K 按通道、V 按 token，少量离群 K 通道保 FP16", "~2.2x（至 4x batch）", "LongBench/GSM8K 近 FP16", "3"],
              ["KVQuant (2304.07189, 2023)", "4bit per-channel + pre-RoPE 量化 + NuQ 非均匀 + 稠密/稀疏离群分离", "~4x", "近无损，撑起 10M 上下文", "2（NuQ 需 LUT）"],
              ["SKVQ (2405.06219, 2024)", "滑窗量化：近期窗口高精度，RNN 式通道重排治 K 离群", "2bit V，1M ctx/80GB", "所述配置下 ppl 追平 FP16", "2（重排=gather，与 paged cache 交互）"],
              ["MiKV (~2402.02617, 2024)", "2bit 量化 + ~5% 离群 token 整行保 FP16", "~5.7x", "纯 2bit 崩溃场景恢复 FP16 水平", "3"],
              ["WKVQuant (2402.12068, 2024)", "W4A4K4V4 联合量化 + 2D 量化 + 参数共享", "4bit KV", "较 W4A4 基线 ppl 小降", "1（耦合权重量化管线）"],
              ["TurboQuant (~2504.19874, 2025)", "在线 per-moment 校准 + 随机旋转/Hadamard 均化离群", "2-4bit", "同位宽持平或超 KIVI", "3（已有同款基础）"],
              ["1-bit KV 长上下文 (2025, id 待核验)", "亚 2bit/三值 KV + 强离群通道处理", "1-1.58bit", "离群处理不足时明显退化", "2（三值需 int 点积核）"],
              ["Massive Activations (2402.17762) + attention sink (2309.17453)", "少数 K 通道巨大激活 + sink 注意力质量是量化误差根因", "使能项", "KIVI/MiKV/sink 保留的精度保证基础", "3（纯切片/掩码）"],
              ["Palu (2407.21118, 2024)", "低秩投影空间内做 KV 量化，注意力中解压", "至 ~90% 消减", "50-70% 比率近无损", "2（离线标定，破坏标准 KV 分页布局）"],
              ["GEAR (2403.05527, 2024)", "2-4bit 量化 + 低秩 + 稀疏矩阵补偿量化误差", "2-4bit + 误差校正", "按构造近无损（LongBench）", "2（三矩阵 per cache，paged 布局复杂）"],
              ["MLA / DeepSeek-V2 (2405.04434, 2024)", "KV 联合低秩投影成 latent（512 维）+ 解耦 RoPE key，只缓存 latent", "93.3% KV 消减", "持平/超 MHA/GQA（预训练期固化）", "1（需架构改造或蒸馏）"],
              ["TransMLA (2502.07864, 2025)", "GQA checkpoint 结构化转换为等价 MLA + 短训练", "~93%", "同 cache 尺寸表达力严格更强", "2（转换训练是成本）"],
              ["CLA (2408.01790, 2024)", "跨层共享 KV 投影，隔层才缓存", "2-8x", "小幅损失，需训练", "1"],
              ["YOCO (2405.05254, 2024)", "自解码器流入单一共享 KV cache，cross-decoder 消费", "KV 只写一次", "512K 预训练验证；只能从头训", "0-1"],
              ["CLLA (2503.03924, 2025)", "低秩 latent + 跨层共享合一设计", "~2.28x 吞吐，cache 至个位数%", "超单独低秩/跨层基线", "1（需训练）"],
          ])
    h(doc, "3.2 选择/合并/架构族（18 篇，全部训练免费、无需自定义 Metal kernel）", 2)
    table(doc,
          ["论文", "方法一句话", "压缩比", "精度影响", "MLX 适配"],
          [
              ["H2O (2306.14048)", "注意力累积分数 heavy-hitter + 近窗淘汰，在线贪心", "至 20x", "20% 预算 <1%；5% 以下多跳退化", "3"],
              ["StreamingLLM (2309.17453)", "sink + 滑窗，窗外全淘汰", "固定 4k 窗跑 4M 流", "ppl 稳但零长程检索（设计使然）——下界地板", "3"],
              ["FastGen (2310.01801)", "按头自适应策略：保留/淘汰/4-8bit 量化按 token 类别", "3.7x 无损，>10x 激进", "LongBench 级近无损", "3"],
              ["Scissorhands (2305.17118)", "重要性持续性：近 H 步重要者继续保留 top-p", "~8x", "8x 近无损（OPT 系）", "3"],
              ["TOVA (2401.06104)", "逐 token 累积注意力幅度决定淘汰，无固定预算", "2-8x 动态", "同内存优于固定预算 H2O", "3"],
              ["SnapKV (2404.14469)", "生成前压缩：观察窗注意力聚类 + 1D 池化稳健选择", "4.5x 内存 2x 解码", "LongBench/NIAH 与 dense 近持平——最强免训练基线", "3"],
              ["PyramidKV (2406.02069)", "层间金字塔预算：低层多留、高层狠压（信息漏斗）", "~12x 近无损", "LongBench 12x 持平 dense", "3（可与逐层预算组合）"],
              ["Quest (2406.10774)", "query 感知页选择：页级 min/max 元数据打分，只读赢页", "~8x 解码稀疏", "RULER/NIAH 近无损；我们的 bucket 质心 top-p 是其更细粒度版", "3"],
              ["D2O (2406.13035)", "层内动态淘汰 + 语义级/token 级区分 + 被淘汰关键 token 跨层补偿", "~10x", "LongBench/RULER 近无损；跨层补偿可直接借入 LSA", "3"],
              ["MiniCache (2405.14366)", "相邻层相似 KV 沿共享方向 SLERP 合并 + 保留项", "~4.8x 峰值", "LongBench 保留 >99%", "2（分组簿记易碎，需细向量化的）"],
              ["KVMerger (2407.08454)", "注意力识别可合并 KV 态，高斯加权自适应合并", "~70% 消减", "~70% 比率 LongBench 无可测损失", "3"],
              ["KeepKV (2504.09936)", "自适应合并 + 等注意力范数约束，消除合并前后分布漂移", "高比率近无损", "专治合并方法的幻觉/一致性下降", "3"],
              ["WeightedKV (2503.01330)", "注意力分数加权 KV 合并（替代均匀平均）", "~50%", "LongBench v1/v2 极小损失", "3"],
              ["GRKV (2605.31105, 2026)", "免训练全局回归做 span 保留，修复淘汰+合并叠加时的边界承载 token 过合并", "SnapKV 级", "长上下文近无损；Ward 聚类+合并会撞同一问题", "3"],
              ["SelKV (2607.16213, 2026)", "逐 token 合并/淘汰决策 + 合并项注意力补偿", "选择性更高有效比率", "长上下文 QA 优于纯合并", "3"],
              ["EchoKV (2611.05903, 2026)", "自校正淘汰：lookahead 工作区，用早期 decode query 抢救后再提交淘汰", "高稀疏", "专治 RULER 分散证据失败模式；多一次 workspace 前向", "3（延迟预算受影响）"],
              ["SSA 分层稀疏注意力 (2610.28441, 2026)", "确定性 top-k + 残余注意力质量的分数引导分层采样", "<5% token 参与注意力", "纯 top-k（Quest/SnapKV 类）崩塌处补齐——高比率下 <1% 要求的最佳单点", "3"],
              ["SparPR (2610.18140, 2026)", "消费级 GPU 稀疏部分重算：query 感知块选择 + 按需重算被淘汰块", "有效比率无上界（淘汰+重算）", "近 dense 精度；唯一可证明恢复精度的族——与 paged SSD 天然协同", "3"],
          ])
    p(doc, "无法核验：CaM（arxiv 检索不到，KVMerger 2407.08454 覆盖同一思路）、DyoQ（零命中，疑 OpenReview-only）。")

    h(doc, "4. 顶尖专家会做什么（四问）", 1)
    h(doc, "4.1 专家会做哪些我们没有要求的事", 2)
    bullets(doc, [
        "先建评测基准而非先做压缩：固定一套 RULER/LongBench/needle 评分脚本与 dense baseline，任何方案先过基准再谈压缩比；"
        "没有 baseline 的压缩比是无意义数字。",
        "做 KV cache 的画像测量：各层、各头、各位置的 KV 范数/离群分布/信息量分布（哪层 KV 可以狠压、哪层动不得），"
        "数据决定方案而不是论文决定方案。",
        "把「注意力质量守恒」当作验收：选中集合上的 softmax 质量、逐层 KL(dense || sparse) 上界，"
        "而不是只看最终答案对错。",
        "考虑 decode 吞吐而不只是显存：压缩后 gather/反量化开销可能吃掉收益，端到端 tokens/s 才是最终指标。",
        "失败模式分析：sink token 漂移、attention sink 丢失、多轮对话 system prompt 被淘汰——每个方案先写死边界。",
    ])
    h(doc, "4.2 专家从哪些最原始素材入手", 2)
    bullets(doc, [
        "模型本体：真实 checkpoint 的逐层 attention 权重与激活（我们已有 Qwen3.8-27B/GLM-5.3-Flash 本地权重），"
        "而不是论文里的二次数据。",
        "原始注意力图：长文档 forward 的 dense attention matrix 导出（按层按头），量化/合并误差直接对 attention 分布度量。",
        "上游论文的实验代码与配置：latent-sparse-attention 仓库、KIVI/Quest 官方实现，读 config 与 cache 布局，"
        "而不是只读论文结论。",
        "硬件计数器：Metal GPU 的带宽/占用率（mlnavigator、Instrument、xctrace），确定瓶颈到底是容量还是带宽。",
        "真实工作负载：fusion 各端（code/cowork/RAG）的实际 prompt 长度分布与命中模式，"
        "以自有 trace 而非合成 benchmark 定目标。",
    ])
    h(doc, "4.3 专家使用的具体方法与工具", 2)
    bullets(doc, [
        "测量：PyTorch hook 导出 KV 统计（我们已建 extract_nlsa_vectors.py 同款管线）；Metal 性能用 xctrace/Metal HUD。",
        "对拍：upstream plain-torch oracle 逐单元对拍（P0 已验证的方法论复用到量化/合并路径）。",
        "快速原型：MLX 侧先做离线仿真（对已导出的真实 KV 做量化/合并/选择，测检索质量），"
        "仿真通过再进引擎，不直接改推理栈。",
        "基准：RULER（needle/multi-key/variable tracking）、LongBench、∞-bench；脚本化进 CI，每次改动自动回归。",
        "统计工具：逐层 KV SVD 谱分析（决定低秩可行域）、按头量化敏感度扫描、Hadamard 旋转后的分布检验。",
    ])
    h(doc, "4.4 除你我之外，什么能证明结果正确", 2)
    bullets(doc, [
        "公开基准复现：RULER/LongBench 分数可被任何人用同 checkpoint 复现——最强外部证据。",
        "与论文报告数字对表：我们的 KIVI 式 4bit 实现跑出的精度应落在原论文报告区间；偏离即实现有 bug。",
        "第三方实现交叉验证：vLLM/llama.cpp/MLX 社区同模型同量化配置的输出对比（同 prompt 同种子 logits KL）。",
        "下游任务不退化：fusion-autotest 的真实 macOS 应用任务、fusion-rag 检索质量、fusion-code 代码任务通过率。",
        "理论界：量化误差对 attention softmax 的扰动上界（softmax 的 Lipschitz 性质），"
        "给「为什么不掉点」一个数学理由而不是经验理由。",
        "持续回归：基准进 CI，任何引擎改动后自动跑，防「当时验证过」的腐化。",
    ])

    h(doc, "5. 优先级建议（可直接尝试）", 1)
    table(doc,
          ["优先级", "方案", "压缩目标", "前置条件", "风险与对策"],
          [
              ["P0-1", "LSA/DSA 稀疏选择全量上线（fusion-lsa，已 P0 完成）",
               "读取侧 4-16x，KV 全存零精度损失", "fusion-mlx 适配器（P2）",
               "边界情形 tie/sink 已对拍；RULER 32k 先行"],
              ["P0-2", "TQ 4bit latent-KV 量化（KIVI/KVQuant 风格离群保护：sink 与异常通道保 fp16、per-channel 分组、Hadamard 旋转）",
               "容量侧 4x", "真实 KV 画像 + 离线仿真", "精度风险中；先 128k RULER 摸底（评审 P0-4）"],
              ["P1-1", "paged SSD 分层 KV（热页统一内存、冷页 SSD）",
               "容量无限扩展，延迟换", "已有 paged-SSD-KV 基础", "随机读延迟；按 bucket 粒度预取"],
              ["P1-2", "GLM-5.3-Flash latent-KV 主线（MLA 类，1M 上下文唯一正解）",
               "10-90x，架构级", "GLM 适配器", "需模型原生支持，Qwen3.8-27B 只做中上下文验证"],
              ["P2-1", "KV 合并（MiniCache/CaM 跨层/跨头合并）",
               "2-5x", "谱分析证明层间相似度", "与 LSA 正交；精度风险中，离线仿真先行"],
              ["P2-2", "2bit/1.58bit KV + 逐层混合位宽（画像驱动）",
               "8-16x", "P0-2 的画像矩阵", "仅低敏感层启用；查询级 fallback"],
          ])
    p(doc, "组合拳（终态）：latent-KV（GLM 路线）+ TQ 4bit + LSA 稀疏选择 + paged SSD = 1M token 在 120GB 内可行，"
           "与迁移方案的内存硬边界一致：Qwen3.8-27B 全量 KV 无法 1M，只做中上下文验证。")
    p(doc, "调研增补的高杠杆点（全部免训练、纯 MLX 算子）：① SSA（2610.28441）残余注意力质量的分数引导采样——"
           "纯 top-k 在高稀疏度崩塌处的补救，直接服务 <1% 损失约束；② D2O（2406.13035）被淘汰关键 token 的跨层补偿——"
           "可借入 LSA 的读取路径；③ SparPR（2610.18140）淘汰后按需重算——与 paged SSD 天然协同"
           "（从 SSD 重算 vs 从权重重算按块决策），是唯一可证明恢复精度的族；④ MiKV 式离群 token FP16 逃生通道"
           "叠加在 TQ 量化路径上；⑤ sink + 离群通道保留（Massive Activations 2402.17762）是量化不掉点的廉价保险。")

    h(doc, "6. 落地路径", 1)
    bullets(doc, [
        "第一步（本周）：真实 KV 画像——27B/GLM 在 16k-128k 文档上的逐层 KV 导出与谱分析（复用 extract 管线）。",
        "第二步：离线仿真架——对导出 KV 施加各压缩法，测 RULER 子集检索质量，产出「每层可压位宽/可合并率」矩阵。",
        "第三步：按优先级实现进 fusion-mlx（TQ 4bit latent-KV 先行），对拍 + 基准回归。",
        "第四步：paged SSD 分层与压缩正交组合，1M 上下文端到端验收。",
    ])

    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    logger.info("wrote %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(build())
