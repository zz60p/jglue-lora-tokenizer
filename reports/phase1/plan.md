# Phase 1 计划（第 1–2 周）
- 完成数据加载与缓存（JNLI/JSQuAD）。
- 输出 EDA（长度分布、标签分布、上下文长度、答案跨度分布）。
- 实现 **基线评测脚手架**：
  - JNLI：多数类 + 词汇重叠启发式。
  - JSQuAD：BM25 句选启发式，计算 EM/F1。
- 编写分词器对比 Demo（fugashi / SudachiPy），为 Phase 3 做铺垫。
- 产出 `results_template.md` 与 `eda_notes.md`，并提交到 GitHub。
