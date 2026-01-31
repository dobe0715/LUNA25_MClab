# LUNA25_MClab

This repository contains the code for the **LUNA25 Challenge** submission.

## 🏆 Leaderboard

You can view the final rankings and evaluation results at the following link:

* **[LUNA25 Closed Testing Phase Leaderboard](https://luna25.grand-challenge.org/evaluation/closed-testing-phase/leaderboard/)**

## 📂 Dataset

This project also utilizes the **LUNA25-MedSAM2** dataset with LUNA25 dataset, which is authorized for use in the LUNA25 competition.

* **Download Link:** [Hugging Face - wanglab/LUNA25-MedSAM2](https://huggingface.co/datasets/wanglab/LUNA25-MedSAM2)

## 🚀 Usage

To train the model for submission, you can use one of the following methods.

### Option 1: Training using torch.cuda.amp

Run the following command to start training with AMP (Automatic Mixed Precision):

```bash
python train_amp.py

```

### Option 2: Training using accelerate library (Recommended)

For multi-GPU or optimized training using Hugging Face Accelerate:

1. **Configure Accelerate** (First time only):
```bash
accelerate config

```


2. **Launch Training**:
```bash
accelerate launch train_accelerate.py

```



## 🔗 References

* **Baseline Code:** [DIAGNijmegen/luna25-baseline-public](https://github.com/DIAGNijmegen/luna25-baseline-public)
