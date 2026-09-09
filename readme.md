# ML From Scratch

Two classic machine learning classification problems, each solved twice: once with `scikit-learn`, and once by implementing the entire model — cost function, gradients, and optimizer — from first principles in NumPy.

The goal wasn't to beat `scikit-learn`. It was to open the black box: to know exactly what `.fit()` is doing internally, and to be able to explain *why* a model converges (or doesn't) rather than just that it runs.

## Projects

| Project | Task | Algorithm | Folder |
|---|---|---|---|
| Rock vs. Mine Prediction | Binary classification of sonar signals | Logistic Regression | [`rock-vs-mine-logistic-regression/`](./rock-vs-mine-logistic-regression) |
| Diabetes Prediction | Binary classification of diagnostic data | Linear SVM | [`diabetes-prediction-svm/`](./diabetes-prediction-svm) |

## What's implemented from scratch (no `sklearn` in the model itself)

- Sigmoid / hinge-loss cost functions
- Gradient and subgradient computation
- Batch gradient descent optimizer
- Stratified train/test split
- Feature standardization (fit on train, applied to test — no data leakage)
- Accuracy scoring
- Convergence tracking and visualization

Each notebook includes a commented-out `scikit-learn` comparison cell at the end to validate the from-scratch results against the library implementation.

## Tech Stack

Python, NumPy, Pandas, Matplotlib, Jupyter Notebook

## Author

Jonathan Felix — Computer Science, Federal University of Technology, Minna
