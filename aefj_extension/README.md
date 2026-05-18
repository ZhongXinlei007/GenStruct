## AEFJ 扩展说明

本目录提供 **Adaptive & Explainable Fusion Judger (AEFJ)** 以及 **全局一致性优化器** 的实现，用于在 OneNet-Co 基础上实现方案一与方案二：

- **自适应权重元学习器**：使用 LLM 对上下文、先验、共现三类信息源进行情境分析，动态输出 `w_context / w_prior / w_cooc`。
- **可解释性报告生成**：利用 LLM 汇总最终判决的证据、权重贡献与全局一致性结果，输出中文判决报告。
- **全局一致性优化**：在文档级别上对所有 mention 进行联合重排，通过 PMI 与文本语义重叠近似 `Compat(c_i, c_j)`，并采用迭代 refinement 获得更稳定的全局解。

### 目录结构

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 模块化入口 |
| `llm_wrapper.py` | 统一的 LLM 加载与调用封装 |
| `meta_learner.py` | AEFJ 情境感知权重元学习器 |
| `explainer.py` | 判决报告生成器（支持 LLM & 模板回退） |
| `global_optimizer.py` | 文档级全局一致性迭代优化器 |
| `aefj_runner.py` | 主运行脚本，串联权重估计、全局优化与解释输出 |

### 快速开始

1. **准备输入**
   - 先运行既有的 Context & Prior linker，得到 `context_pred_num`、`prior_pred_num`、`context_confidence`、`prior_confidence`、`context_response`、`prior_response` 等字段。
   - 额外提供 `context_entities`（文档内其余实体的 wiki id 列表）以便共现/一致性建模。

2. **构建 PMI 表（可选）**
   ```bash
   python new_code/build_cooccurrence.py --input <processed_docs> --output <pmi.json>
   ```

3. **运行 AEFJ**
   ```bash
   python -m aefj_extension.aefj_runner \
     --dataset_name ace2004 \
     --input_path result/ace2004_mfj.jsonl \
     --output_path result/ace2004_aefj.jsonl \
     --meta_model_name glm-4-api \
     --meta_model_path dummy \
     --share_model \
     --cooc_pmi_path data/cooc_pmi.json \
     --doc_field doc_id \
     --enable_global \
     --top_k 3 \
     --coherence_weight 0.35
   ```
   关键参数：
   - `--meta_model_*`：元学习器所用 LLM，可与解释生成共享 (`--share_model`)。
   - `--disable_meta`：关闭 LLM 元学习，直接使用 `--meta_default_weights`。
   - `--enable_global`：激活方案二的全局一致性优化；`--doc_field` 指定文档编号字段。
   - `--context_entities_field`：上下文实体字段名称（默认 `context_entities`）。

4. **输出字段**
   - `aefj_weights` / `aefj_scores`：每条 mention 的动态权重与候选得分。
   - `aefj_ranked_candidates`：Top-K 候选及其融合得分。
   - `global_coherence_bonus` / `global_coherence_note`：文档级协同收益与解释。
   - `aefj_explanation`：LLM 生成的最终判决报告。
   - `final_pred_num` / `final_pred_entity`：AEFJ + Global 的最终结果。

### 实验建议

1. **对比实验**
   - MFJ (静态权重) vs AEFJ (动态权重)。
   - AEFJ (无全局) vs AEFJ + Global（`--enable_global`）。

2. **消融设置**
   - 仅打开 Context/Prior/Co-occ 权重，观察元学习器的自适应性。
   - 调整 `coherence_weight`、`top_k` 与 `coherence_iter` 评估全局优化稳定性。

3. **效率注意事项**
   - 元学习器与解释生成均调用 LLM，建议使用低温 (temperature=0) 并开启复用。
   - 大规模数据可先关闭 `--enable_global` 或调小 `top_k` 加速。

### 依赖

与 OneNet-Co 相同，额外依赖仅为可访问的 LLM 模型（本地或 API）。若使用 API（如 `glm-4-api`），需确保相关密钥已通过环境变量配置。

### 常见问题

1. **没有文档 ID 怎么办？**  
   使用 `--doc_field ""`，系统会默认每个 mention 自成文档（全局优化自动跳过）。

2. **共现矩阵缺失**  
   `cooc_pmi_path` 可为空；此时 AEFJ 仍可运行，只是 `w_cooc` 会因缺乏证据而降低。

3. **LLM 输出非 JSON**  
   元学习器内置容错，会自动 fallback 到默认权重并记录 `aefj_meta_reason`。


