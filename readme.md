# Towards Automatic Evaluation and High-Quality Pseudo-Parallel Dataset Construction for Audio Editing: A Human-in-the-Loop Method

**[arXiv Link to be added]**

---

## Model Structure

![Model Structure](img/model.png)

---

## Overview

**Subjective Audio Editing Evaluation Dataset Construction**  
We construct **AuditScore**, the first comprehensive benchmark for subjective audio editing evaluation, consisting of over 6,300 edited samples from 7+ representative frameworks and 23+ system variants. Each sample is annotated by professional raters on three key aspects: **Quality**, **Relevance**, and **Faithfulness**.

**Automatic MOS Prediction**  
Based on AuditScore, we train **AuditEval**, the first model tailored for automatic MOS-style scoring in audio editing. It effectively mitigates the lack of objective metrics and the high cost of human evaluation.

---

## Dataset: AuditScore

AuditScore is a human-annotated dataset comprising **6,360 pairs of original and edited audio samples**, totaling 35.3 hours in duration. These samples are generated using **23 distinct system configurations** across seven representative audio editing approaches.  

**Dataset Figures:**  

<p float="left">
  <img src="img/q_.png" width="30%" />
  <img src="img/r_.png" width="30%" />
  <img src="img/f_.png" width="30%" />
</p>
<p float="left">
  <img src="img/q.png" width="30%" />
  <img src="img/r.png" width="30%" />
  <img src="img/f.png" width="30%" />
</p>

**[Dataset Download Link to be added]**

---

## Model Checkpoints

- **Pretrained CLAP**: [https://huggingface.co/lukewys/laion_clap](https://huggingface.co/lukewys/laion_clap)  
- **AuditEval Quality MLP**: [Link to be added]  
- **AuditEval Relevance MLP**: [Link to be added]  
- **AuditEval Faithfulness MLP**: [Link to be added]  

---

## 使用 AuditScore Mos-Predictor 进行自动评测

本节介绍如何使用 AuditScore 提供的 MOS-predictor 工具对音频编辑结果进行自动评分：

1. **配置输入**  
   在 `pred.py` 中指定：
   - 原始音频路径 `ori_path`
   - 编辑后的目标音频路径 `tar_path`
   - 原始文本 `ori_text`
   - 编辑指令文本 `tar_text`

2. **运行评测**  
   执行以下命令即可获取对应音频的三个评分（Quality、Relevance、Faithfulness）：
   ```bash
   bash pred.sh
   ```

3. **训练模型**  
   如果需要训练或微调 MOS-predictor 模型，可以使用：
   ```bash
   bash train.sh
   ```

以上步骤可快速完成音频编辑效果的自动化评估，减少人工打分成本。

---

## Acknowledgements

We thank the authors of **[MusicEval Baseline](https://github.com/NKU-HLT/MusicEval-baseline)** for their valuable reference and resources.
