# fusion-lsa

Latent Sparse Attention（LSA）的 MLX/Metal 移植，属 [Fusion](../) 本地优先 Apple Silicon AI 生态。

LSA 不改 attention kernel，只决定「attention 读哪些 key」：index key 先池化再用 Ward 聚类成
bucket，query 对 bucket 质心打分，取 top-p 个 bucket 展开成 top-k key 集合，另加 sink token 与
滑动窗口。本仓库移植的是选择逻辑（非 CUDA kernel），源自
[latent-sparse-attention](https://github.com/Prefrontal-AI/latent-sparse-attention) 的 plain-torch
参考实现，并以其为 oracle 做一致性对拍。

状态：P0（算法移植 + 对拍 + 成本闸门）完成，P1（打包、map 转换器、checkpoint 端到端）进行中。
详见 `docs/p0-report.md`。

## P0 结果（M5 Max 128GB）

- bucket 划分、成员、大小、选中 key 集合与 oracle 完全一致；质心误差 ≤ 1.5e-4（fp32 求和顺序）。
- decode 选择 ~0.7-1.0 ms/query；ward/bucket 构建 7.2s / 128k 上下文（仅页边界执行）。
- 128k 端到端：needle 命中，选中集合持有 > 0.9 的 attention softmax 质量。

## 目录结构

```
src/fusion_lsa/
  core/         ward.py, reference.py — oracle 选择逻辑的 MLX 移植
  evaluation/   selection_bench.py — 分块选择成本 bench
tests/          oracle 对拍 + 128k 端到端
docs/           P0 报告、bench 结果与日志
```

## 环境准备

```bash
cd /Users/dahai/fusion
source .venv/bin/activate
cd fusion-lsa && pip install -e ".[test]"
python -m pytest tests/
```

对拍测试从 `~/model/latent-sparse-attention/src` 读取上游 oracle（可用 `LSA_UPSTREAM` 覆盖），
不存在时自动跳过。

## 硬边界

- 只移植选择逻辑：attention 本身是对选中 key 的标准 SDPA。
- 1M+ 上下文需要 latent-KV 架构 + KV 量化；full-KV 模型在 120GB 内存无法跑到 1M。
- bucket/hierarchy 只在页边界增长；decode 每 token 只做选择，绝不重新聚类。
