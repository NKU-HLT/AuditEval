# Towards Automatic Evaluation and High-Quality Pseudo-Parallel Dataset Construction for Audio Editing: A Human-in-the-Loop Method
[ArXiv Paper](https://arxiv.org/pdf/2508.11966)

## 🧩 Abstract

Audio editing aims to manipulate audio content based on textual descriptions, supporting tasks such as adding, removing, or replacing audio events. Despite recent progress, the lack of high-quality benchmark datasets and comprehensive evaluation metrics remains a major challenge for both assessing audio editing quality and improving the task itself.  
In this work, we propose a novel approach for audio editing tasks by incorporating expert knowledge into both the evaluation and dataset construction processes:

1. **AuditScore**: The first comprehensive dataset for subjective evaluation of audio editing, consisting of over 6,300 edited samples generated from 7 representative audio editing frameworks and 23 system configurations.
2. **AuditEval**: A model designed for automatic MOS-style scoring tailored to audio editing tasks.
3. **Pseudo-Parallel Dataset**: Using AuditEval, we evaluate and filter a large amount of synthetically mixed editing pairs, constructing a high-quality pseudo-parallel dataset.

Objective experiments validate the effectiveness of our expert-informed filtering strategy in yielding higher-quality data while also revealing the limitations of relying solely on objective metrics.

![Model Structure](img/model.png)

## 🛠️ Overview

**Subjective Audio Editing Evaluation Dataset Construction**  
We construct **AuditScore**, the first comprehensive benchmark for subjective audio editing evaluation, consisting of over 6,300 edited samples from 7+ representative frameworks and 23+ system variants. Each sample is annotated by professional raters on three key aspects: **Quality**, **Relevance**, and **Faithfulness**.

**Automatic MOS Prediction**  
Based on AuditScore, we train **AuditEval**, the first model tailored for automatic MOS-style scoring in audio editing. It effectively mitigates the lack of objective metrics and the high cost of human evaluation.


## 📚 Dataset: AuditScore

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

**[Dataset Download Link will be added after acceptance]**

## 🧠 Model Checkpoints

- **Pretrained CLAP**: [https://huggingface.co/lukewys/laion_clap](https://huggingface.co/lukewys/laion_clap)  
- **AuditEval Quality MLP**: [quality_epoch_50.pt](quality_epoch_50.pt)  
- **AuditEval Relevance MLP**: [relevance_epoch_30.pt](relevance_epoch_30.pt)  
- **AuditEval Faithfulness MLP**: [faithfulness_epoch_35.pt](faithfulness_epoch_35.pt)


## 🚀 Using AuditScore MOS-Predictor for Automatic Evaluation

This section describes how to use the **AuditScore MOS-predictor** tool to automatically evaluate audio editing results:

1. **Configure Inputs**  
   In `pred.py`, specify:
   - Original audio path `ori_path`
   - Edited target audio path `tar_path`
   - Original transcript `ori_text`
   - Editing instruction text `tar_text`

2. **Run Evaluation**  
   Execute the following command to obtain the three scores (Quality, Relevance, Faithfulness) for the audio:
   ```bash
   bash pred.sh
   ```

3. **Train the Model**  
   If you need to train or fine-tune the MOS-predictor, use:
   ```bash
   bash train.sh
   ```

These steps allow quick automatic assessment of audio editing performance, reducing the need for costly human annotations.


## 📜 Acknowledgment

This project is greatly supported by prior works such as **[ZETA](https://arxiv.org/abs/2402.10009)** and **[MusicEval](https://arxiv.org/pdf/2501.10811)**.


## 🤝🏻 Contact  

Should you have any questions, please contact  
📧 **2120240729@mail.nankai.edu.cn**

## 🧩 Citation

If you find this tool helpful, please cite both **ZETA** and **MusicEval**.

```bibtex
@misc{jia2025automaticevaluationhighqualitypseudoparallel,
      title={Towards Automatic Evaluation and High-Quality Pseudo-Parallel Dataset Construction for Audio Editing: A Human-in-the-Loop Method}, 
      author={Yuhang Jia and Hui Wang and Xin Nie and Yujie Guo and Lianru Gao and Yong Qin},
      year={2025},
      eprint={2508.11966},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2508.11966}, 
}
```

