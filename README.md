# DA5401 Data Challenge: Metric Learning & Domain Adaptation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Project Overview

This repository contains the complete solution for the **DA5401 Data Challenge** focused on automating the evaluation of Large Language Model (LLM) responses through metric learning and domain adaptation techniques.

**Problem:** Predict a "fitness score" (0-10) indicating how well an LLM response satisfies a specific evaluation metric in the context of given prompts.

**Final Result:** RMSE of **0.82** (17.2% improvement over baseline)

---

## 🎯 Problem Statement

### Challenge Description
- **Task:** Metric learning for LLM response evaluation
- **Goal:** Predict fitness scores (0-10) for LLM responses
- **Evaluation:** Root Mean Squared Error (RMSE)

### Dataset Characteristics
- **Training Samples:** 5,000
- **Test Samples:** 3,638
- **Unique Metrics:** 145 across 50 categories
- **Languages:** Tamil, Hindi, Assamese, Bengali, Bodo, Sindhi, English
- **Key Challenge:** Distribution shift between training (mean=9.12) and test data (mean≈6.1)

---

## 🚀 Solution Approach

Our winning strategy combines three key pillars:

### 1. Multilingual Embeddings
- **Model:** `intfloat/multilingual-e5-large`
- **Embeddings:** 1024-dimensional dense vectors
- **Training:** Contrastive learning on 100+ languages
- **Advantage:** Robust semantic understanding across diverse languages

### 2. Rich Feature Engineering
Created **4097-dimensional feature vectors** comprising:
- Metric embeddings (1024 dims)
- Text embeddings (1024 dims)
- Absolute difference (1024 dims)
- Element-wise product (1024 dims)
- Cosine similarity (1 dim)

### 3. Ensemble + Calibration
- **Models:** LightGBM + XGBoost ensemble
- **Augmentation:** 15,000 synthetic negative samples (scores 0-3)
- **Calibration:** Quantile-based mapping to handle distribution shift
- **Cross-Validation:** 5-fold stratified CV

---

## 📁 Repository Structure

```
da5401-2025-data-challenge/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
│
├── data/                                  # Dataset (not included in repo)
│   ├── train.csv
│   └── test.csv
│
├── augmentation_strategy/                 # Data augmentation pipeline
│   ├── 1_analysis/
│   │   └── 1_distribution_analysis.ipynb # Preliminary EDA
│   └── 2_quick_augmentation/
│       └── 2_quick_augmentation.ipynb    # Synthetic data generation
│
├── winning_strategy_calibrated.ipynb     # Main model training notebook
├── comprehensive_analysis_and_modeling.ipynb # Complete pipeline
│
├── Report/                                # LaTeX report & visualizations
│   ├── Kaggle Report.tex                 # Main report (687 lines)
│   ├── images/                           # All figures (11 images)
│   │   ├── score_distribution.png
│   │   ├── distribution_gap.png
│   │   ├── distribution_raincloud.png
│   │   ├── metric_treemap.png
│   │   ├── metric_categories.png
│   │   ├── metric_frequency.png
│   │   ├── text_lengths.png
│   │   ├── embedding_space.png
│   │   ├── embedding_tsne_simulation.png
│   │   ├── model_comparison.png
│   │   └── model_performance_ladder.png
│   │
│   └── image_generation_scripts/         # Scripts to regenerate all images
│       ├── README.md
│       ├── requirements.txt
│       ├── generate_all_images.py        # Master script
│       ├── 1_score_distribution.py
│       ├── 2_metric_treemap.py
│       ├── 3_text_lengths.py
│       ├── 4_embedding_tsne.py
│       └── 5_model_performance.py
│
└── models/                                # Trained models (not in repo)
    ├── lgbm_model.pkl
    └── xgb_model.pkl
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) GPU for faster embedding generation

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd da5401-2025-data-challenge
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - ML utilities
- `lightgbm` - Gradient boosting
- `xgboost` - Gradient boosting
- `sentence-transformers` - E5 embeddings
- `matplotlib`, `seaborn` - Visualization

### Step 3: Download Data
Place your data files in the `data/` directory:
```
data/
├── train.csv
└── test.csv
```

---

## 💻 Usage

### Quick Start: Run Complete Pipeline

```bash
jupyter notebook comprehensive_analysis_and_modeling.ipynb
```

This notebook includes:
1. Exploratory Data Analysis
2. Feature Engineering
3. Model Training
4. Validation
5. Calibration

### Step-by-Step Execution

#### 1. Preliminary Analysis
```bash
jupyter notebook augmentation_strategy/1_analysis/1_distribution_analysis.ipynb
```

#### 2. Data Augmentation
```bash
jupyter notebook augmentation_strategy/2_quick_augmentation/2_quick_augmentation.ipynb
```

#### 3. Model Training
```bash
jupyter notebook winning_strategy_calibrated.ipynb
```

### Generate Report Images
```bash
cd Report/image_generation_scripts
pip install -r requirements.txt
python generate_all_images.py
```

### Compile LaTeX Report
```bash
cd Report
pdflatex "Kaggle Report.tex"
```

---

## 📈 Results

### Model Performance

| Model | Approach | CV RMSE | Improvement |
|-------|----------|---------|-------------|
| Baseline | Mean Prediction | 0.99 | - |
| TF-IDF + Ridge | Classical ML | 0.92 | 7.1% |
| BERT + MLP | Deep Learning | 0.88 | 11.1% |
| **E5 + Ensemble** | **Our Approach** | **0.82** | **17.2%** |

### Key Achievements
- ✅ 17.2% improvement over baseline
- ✅ 6.8% better than BERT-based deep learning
- ✅ Effective multilingual handling
- ✅ Robust to distribution shift
- ✅ Interpretable feature importance

---

## 🔑 Key Features

### 1. Contrastive Learning
Leverages E5's contrastive pre-training:
```
L_contrastive = -log[exp(sim(q,p+)/τ) / (exp(sim(q,p+)/τ) + Σ exp(sim(q,pi-)/τ))]
```

### 2. Feature Engineering
- **Absolute Difference:** Captures semantic distance
- **Element-wise Product:** Highlights aligned features
- **Cosine Similarity:** Global semantic alignment

### 3. Synthetic Negatives
Generated 15,000 samples with scores 0-3 to balance distribution:
- Random pairing of embeddings
- Gaussian noise injection (σ=0.6)
- Metric swapping

### 4. Quantile Calibration
Maps predictions to target distribution:
```
score_calibrated(i) = Φ^(-1)(rank(i)/(n+1); μ=6.1, σ=2.5)
```

---

## 📚 Key Files Description

### Notebooks

#### `comprehensive_analysis_and_modeling.ipynb`
**Main notebook** containing the complete pipeline:
- EDA with all visualizations
- Feature engineering
- E5 embedding generation
- Model training (LightGBM + XGBoost)
- Cross-validation
- Calibration

#### `winning_strategy_calibrated.ipynb`
**Winning model** implementation:
- Final model architecture
- Hyperparameter tuning
- Calibration strategy
- Prediction generation

#### `augmentation_strategy/1_analysis/1_distribution_analysis.ipynb`
**Preliminary analysis:**
- Score distribution analysis
- Metric diversity exploration
- Text length patterns
- Language detection
- Correlation analysis

#### `augmentation_strategy/2_quick_augmentation/2_quick_augmentation.ipynb`
**Data augmentation:**
- Synthetic negative generation
- Three augmentation strategies
- Distribution balancing

### Report

#### `Report/Kaggle Report.tex`
**Comprehensive LaTeX report (687 lines, 11 figures):**
- Problem statement
- Sentence transformer selection (E5)
- Exploratory data analysis
- Feature engineering details
- Model architecture (TikZ diagram)
- Results & comparisons
- Conclusion

---

## 🎨 Visualizations

All visualizations are generated programmatically and included in the report:

1. **Score Distribution** (4-panel): Histogram, box plot, rounded bars, CDF
2. **Distribution Gap**: Train vs test distribution shift
3. **Distribution Raincloud**: Density comparison
4. **Metric Treemap**: 145 metrics across 50 categories
5. **Metric Categories**: Bar chart of top categories
6. **Metric Frequency**: Long-tail distribution
7. **Text Lengths**: 3-panel comparison (train vs test)
8. **Embedding Space**: 3D visualization
9. **t-SNE Embeddings**: 2D projection colored by score
10. **Model Comparison**: Bar chart of RMSE
11. **Performance Ladder**: Improvement percentages

---

## 🧪 Experiments & Ablations

### Models Tested
- ❌ Baseline (mean prediction): 0.99 RMSE
- ❌ TF-IDF + Ridge: 0.92 RMSE
- ❌ BERT + MLP: 0.88 RMSE (overfitting issues)
- ✅ E5 + Ensemble: 0.82 RMSE

### Why Tree-Based Models Won
1. **Tabular efficiency**: Better on structured features
2. **Limited data**: 5K samples insufficient for neural networks
3. **No normalization needed**: Handles mixed scales naturally
4. **Feature interactions**: Captures complex patterns automatically
5. **Interpretability**: Clear feature importance

### Ablation Studies
- Without synthetic negatives: +0.05 RMSE
- Without calibration: +0.08 RMSE
- Single model (no ensemble): +0.03 RMSE
- Without interaction features: +0.04 RMSE

---

## 📝 Technical Details

### Embeddings
- **Model:** intfloat/multilingual-e5-large
- **Dimensions:** 1024
- **Input packing:** `[SYS] {system} [USR] {user} [RES] {response}`
- **Batch size:** 32
- **Device:** CUDA if available, else CPU

### Models
**LightGBM:**
- Learning rate: 0.05
- Num leaves: 31
- Max depth: -1 (unlimited)
- Min data in leaf: 20
- Bagging fraction: 0.8

**XGBoost:**
- Learning rate: 0.05
- Max depth: 6
- Min child weight: 1
- Subsample: 0.8
- Colsample bytree: 0.8

### Cross-Validation
- **Strategy:** 5-fold stratified (by score bins)
- **Metric:** RMSE
- **Final model:** Average of LightGBM + XGBoost predictions

---

## 🤝 Contributing

This is an academic project for DA5401 Data Analytics Laboratory. However, suggestions and improvements are welcome!

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Naveen US**  
Student ID: DA25M020  
MTech in Data Science  
IIT Madras

---

## 🙏 Acknowledgments

- **Course:** DA5401 - Data Analytics Laboratory
- **Institution:** IIT Madras
- **Semester:** 1, 2024-25
- **Model:** intfloat/multilingual-e5-large (Hugging Face)
- **Libraries:** scikit-learn, LightGBM, XGBoost, sentence-transformers

---

## 📞 Contact

For questions or discussions about this project:
- Email: [Your email if you want to include]
- GitHub: [Your GitHub profile]

---

## 🔗 Quick Links

- [Report PDF](Report/Kaggle%20Report.pdf) (after compilation)
- [Image Generation Scripts](Report/image_generation_scripts/)
- [Main Notebook](comprehensive_analysis_and_modeling.ipynb)
- [Winning Strategy](winning_strategy_calibrated.ipynb)

---

**Last Updated:** November 2024

**Status:** ✅ Complete & Ready for Submission

