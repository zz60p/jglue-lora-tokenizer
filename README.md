# JGLUE-LoRA-Tokenizer
**Project 1: 日语 JGLUE 统一轻量微调（LoRA）与分词器影响研究**  
（从 0 起步，Colab/Kaggle 可跑；阶段一包含 EDA/评测脚手架与启用脚本）

> 目标：在 JGLUE（以 **JNLI** / **JSQuAD** 为起点）上，基于 3–7B 级别日文 LLM 的 **QLoRA/LoRA** 轻量微调，系统比较 **MeCab/fugashi、SudachiPy、SentencePiece** 对性能与可解释性的影响，并最终开源复现实验与报告。

## 阶段划分（建议 10–12 周）
- **Phase 0（2025-09-01 ± 3 天）**：环境与仓库初始化（你现在下载的就是此模板）。
- **Phase 1（第 1–2 周）**：数据加载 + EDA + 评测脚手架 + 启动基线（不含微调）。
- **Phase 2（第 3–4 周）**：单任务 LoRA 微调（JNLI/JSQuAD），记录显存/速度/指标。
- **Phase 3（第 5–6 周）**：分词器对比 + JSQuAD 答案跨度校准。
- **Phase 4（第 7–8 周）**：多任务统一微调 + 交叉泛化评测。
- **Phase 5（第 9–10 周）**：报告撰写、表格与复现脚本、结果可视化。

## 快速开始（Colab 一键）
1. 把本仓库上传到你的 GitHub（或直接解压到 Colab 工作目录）。  
2. 在 Colab 执行：
```bash
!git clone <你的仓库URL> jglue-lora-tokenizer
%cd jglue-lora-tokenizer
!bash scripts/setup_colab.sh
```
3. 运行阶段一：
```bash
# 数据下载与缓存检查
!python scripts/download_datasets.py

# JNLI 规则/多数类基线 + 评测
!python src/eval/jnli_eval.py --split validation

# JSQuAD 抽取式（BM25+句选）启发式基线 + 评测
!python src/eval/jsquad_eval.py --split validation
```
4. 把 `reports/phase1/` 的输出、图表与日志推送回 GitHub。

## 目录结构
```
jglue-lora-tokenizer/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt
├─ scripts/
│  ├─ setup_colab.sh
│  ├─ download_datasets.py
├─ src/
│  ├─ data/
│  │  ├─ load_jglue.py
│  ├─ eval/
│  │  ├─ metrics.py
│  │  ├─ jnli_eval.py
│  │  └─ jsquad_eval.py
│  ├─ tokenization/
│  │  ├─ mecab_fugashi_demo.py
│  │  └─ sudachi_demo.py
├─ notebooks/
│  ├─ 01_data_checks.ipynb
│  └─ 02_tokenizer_probing.ipynb
├─ reports/
│  └─ phase1/
│     ├─ plan.md
│     ├─ eda_notes.md
│     └─ results_template.md
└─ configs/
   ├─ task_jnli.yaml
   └─ task_jsquad.yaml
```

## 免责声明
- 本仓库仅提供研究用途的脚手架；请遵循各数据集/模型的许可证。
- 后续阶段（LoRA/QLoRA）将在 Phase 2 以后执行。

