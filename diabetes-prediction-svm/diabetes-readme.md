# Diabetes Prediction — Linear SVM From Scratch

Classifies patients as diabetic or non-diabetic from diagnostic measurements, using a linear-kernel Support Vector Machine built entirely from first principles in NumPy — no `sklearn.svm`.

## Dataset

The [PIMA Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database): 768 samples, 8 diagnostic features (glucose, BMI, age, etc.), labeled diabetic (1) or non-diabetic (0).

## What's built from scratch

- **Hinge-loss objective** — the soft-margin SVM cost function, $\frac{1}{2}\|w\|^2 + C \cdot \text{avg hinge loss}$
- **Subgradient computation** — gradients only come from points that violate the margin, which is the part of SVM math that's easiest to gloss over when only calling `.fit()`
- **Batch subgradient descent** — the training loop that steps `w` and `b`
- **Stratified train/test split**
- **Feature standardization** — a from-scratch equivalent of `StandardScaler`, fit on train only
- **Accuracy scoring**
- **Cost history tracking + convergence plot**

## Results

Achieved test accuracy comparable to `scikit-learn`'s `SVC(kernel='linear')` on the same split — though not necessarily identical, since `SVC` solves the dual optimization problem exactly, while this implementation approximates it via gradient descent. See the notebook's output cells and the commented-out `sklearn` comparison cell for exact numbers.

## How to Run

1. Open `Diabetes_Prediction_FromScratch.ipynb` in Jupyter or Google Colab.
2. Upload `diabetes.csv` to the same environment (in Colab: to `/content/`).
3. Run the cells top to bottom.
4. Experiment: change `C` (regularization strength), `learning_rate`, and `num_iterations` in the training cell and re-run to see how they affect convergence and test accuracy.

## Tech Stack

Python, NumPy, Pandas, Matplotlib
