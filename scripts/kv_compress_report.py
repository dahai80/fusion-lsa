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
    h(doc, "3.1 量化与 latent/低秩族", 2)
    p(doc, PLACEHOLDER_QUANT if "PLACEHOLDER_QUANT" in globals() else "（调研填充）")
    h(doc, "3.2 选择/合并/架构族", 2)
    p(doc, PLACEHOLDER_EVICT if "PLACEHOLDER_EVICT" in globals() else "（调研填充）")

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
    p(doc, PLACEHOLDER_PRIO if "PLACEHOLDER_PRIO" in globals() else "（调研填充）")

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
