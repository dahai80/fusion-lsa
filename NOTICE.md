# NOTICE

This product includes code derived from **Latent Sparse Attention**
(Prefrontal-AI/latent-sparse-attention), licensed under the Apache License,
Version 2.0: <https://github.com/Prefrontal-AI/latent-sparse-attention>.

Upstream copyright notice, retained as required by Apache-2.0 section 4(d):

    Copyright 2026 Prefrontal AI

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Upstream publication:

    Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention
    https://arxiv.org/abs/2610.42147

This repository ports the upstream selection logic (not the CUDA kernels) to
MLX/Metal and is parity-tested against the upstream implementation as an
oracle.

Scope note: the upstream PaperTrail kit is licensed CC BY 4.0 and is NOT
included in this distribution.
