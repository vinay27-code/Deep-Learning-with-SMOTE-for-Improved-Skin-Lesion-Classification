# 🧠 Deep Learning with SMOTE for Improved Skin Lesion Classification

A deep learning project aimed at accurately classifying dermatoscopic skin lesion images into **benign** or **malignant** categories using Convolutional Neural Networks (CNN) and class-balancing techniques like **SMOTE** (Synthetic Minority Over-sampling Technique). This project addresses data imbalance — a common challenge in medical datasets — to improve generalization and model performance.

## 🎯 Objective

To enhance the classification accuracy of skin lesions by:
- Applying SMOTE to oversample the minority class
- Training a CNN on the balanced dataset
- Evaluating performance through accuracy, recall, precision, F1-score, and AUC

---

## 📁 Dataset

- **Source**: Publicly available **HAM10000** dataset (or similar)
- **Size**: 10,000+ dermatoscopic images
- **Labels**: Binary classification - `benign`, `malignant`

---

## 🛠️ Technologies & Tools

- Python 3.x
- TensorFlow / Keras
- Scikit-learn
- NumPy, Pandas, Matplotlib
- SMOTE from `imblearn.over_sampling`
- Google Colab / Jupyter Notebook

---

## 🔍 Key Features

- 🧪 **CNN architecture** tailored for image classification
- ⚖️ **SMOTE** to balance the dataset and avoid bias
- 📈 Performance metrics:
  - Accuracy
  - Precision / Recall / F1-Score
  - Confusion Matrix
  - ROC Curve and AUC

---

## 📊 Results Summary

- **Balanced data** led to significant improvement in:
  - **Recall for minority class (~35%)**
  - **Validation accuracy (~20%)**
- Final **F1-Score**: `~92%`
- AUC: `High`, indicating strong classification capability

---

## 🧪 Model Pipeline

1. Data preprocessing and image augmentation
2. Application of SMOTE to training set
3. CNN model training and tuning
4. Evaluation on test set
5. ROC/AUC analysis and visualization

---

## 📌 How to Run

1. Clone the repository:
   ```bash
    git clone https://github.com/vinay27-code/


2. Install dependencies:

pip install -r requirements.txt


3. Run the notebook:

   jupyter notebook Batch5MininProject.ipynb



📈 Sample Visualizations:
📉 ROC Curve
![image](https://github.com/user-attachments/assets/df5572b3-2f03-4726-913b-edb8a31c4d4f)




📊 Confusion Matrix
![image](https://github.com/user-attachments/assets/df71f7b8-c8b4-418a-b72c-5ec9b301cead)
![image](https://github.com/user-attachments/assets/27e74b5e-5206-4c47-91dc-df07572f27f1)

   

📄 License
This project is licensed under the MIT License.

👤 Author
Made with ❤️ by Vinay

If this project helped you, drop a ⭐, share, or fork the repo!   
