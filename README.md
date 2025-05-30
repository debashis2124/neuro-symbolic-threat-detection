# Neuro-Symbolic Threat Detection Pipeline

This project implements a hybrid threat detection system that fuses LSTM-based neural anomaly detection with symbolic rule reasoning for enhanced cybersecurity analytics.


This repository contains a complete neuro-symbolic threat detection pipeline. It integrates:

Neural anomaly detection using LSTM Autoencoders
Symbolic scoring via handcrafted security rules
Hybrid fusion of reasoning signals
Augmentation with synthetic and adversarial attacks
Transfer learning via fine-tuning
Classifier-based threat categorization and visual evaluation

## Structure

- `data/`: Dataset loading functions
- `preprocessing/`: Feature scaling and data splitting
- `autoencoder/`: LSTM-based reconstruction model
- `symbolic_rules/`: Symbolic scoring rules
- `hybrid_fusion/`: Fusion of symbolic + neural scores
- `ml_classifier/`: MLP training and evaluation
- `augmentation/`: Synthetic attack generation
- `gan_training/`: GAN training for adversarial samples
- `transfer_learning/`: Autoencoder fine-tuning
- `evaluation/`: Results consolidation
- `visualization/`: Figure plotting
- `main.py`: Complete pipeline execution script
- `one in all.py`: Merged code for execution script within one file (colab)
## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Requirements

- Python 3.8+
- TensorFlow, scikit-learn, pandas, numpy, matplotlib, seaborn