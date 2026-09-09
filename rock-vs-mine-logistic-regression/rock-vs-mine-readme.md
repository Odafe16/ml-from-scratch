# Rock vs. Mine Prediction — Logistic Regression From Scratch

Classifies sonar signal readings as either a **rock** or a **mine**, using a logistic regression classifier built entirely from first principles in NumPy — no `sklearn.linear_model`.

## Dataset

The [Sonar dataset](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks): 208 samples, 60 numerical features per sample (sonar signal energy at different angles), labeled `R` (rock) or `M` (mine).

## What's built from scratch

- **Sigmoid function** — squashes the linear combination of inputs into a probability
- **Binary cross-entropy cost function** — measures how wrong the predicted probabilities are
- **Gradient computation** — the analytical derivative of the cost function with respect to `w` and `b`
- **Batch gradient descent** — the training loop that repeatedly steps `w` and `b` against the gradient
- **Stratified train/test split** — preserves the rock/mine ratio in both sets
- **Feature standardization** — fit on the training set only, applied to test data and new predictions
- **Accuracy scoring**
- **Cost history tracking + convergence plot**

## Results

Achieved test accuracy comparable to `scikit-learn`'s `LogisticRegression` on the same split (see the notebook's output cells and the commented-out `sklearn` comparison cell at the end for exact numbers).

## How to Run

1. Open `Rock_vs_Mine_Prediction_FromScratch.ipynb` in Jupyter or Google Colab.
2. Upload `sonar_data.csv` to the same environment (in Colab: to `/content/`).
3. Run the cells top to bottom — each one is a self-contained building block (sigmoid → cost → gradients → training loop → evaluation).
4. Experiment: change `learning_rate` and `num_iterations` in the training cell and re-run to see how convergence changes.

## Tech Stack

Python, NumPy, Pandas, Matplotlib
