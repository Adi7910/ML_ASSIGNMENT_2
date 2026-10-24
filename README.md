# ML_ASSIGNMENT_2

This project studies face-identity classification with the Olivetti Faces dataset. The experiments compare a standalone decision tree with two ensemble approaches—bootstrap aggregation (bagging) and multiclass AdaBoost using SAMME. The notebooks also examine how masking part of a face and changing the training sample affect model performance.

## Experiment overview

- **Dataset:** 400 grayscale face images, 40 identities, each image represented by 4,096 features (a flattened 64 × 64 image).
- **Data split:** 240 training images, 80 validation images, and 80 test images.
- **Corrupted test set:** a copy of the test images with the central 16 × 16 pixel region (rows and columns 24–39) set to zero.
- **Main methods:** one decision tree, bagging implemented in the notebook, and multiclass SAMME boosting implemented in the notebook.
- **Reproducibility:** the notebook derives its base random seed from the roll-number string `M26CPS001`.

## Files in this project

```text
.
├── Split(1).ipynb       # Loads the dataset, inspects class counts, and prepares/loads splits
├── Ans1_3(3).ipynb      # Experiments for the tree, bagging, boosting, and bias–variance questions
├── person_data.npz      # Saved train/validation/test indices (required by the main notebook)
└── README.md
```

The results directories and figures are written by notebook cells when those cells are executed. Their exact contents depend on which cells have been run and saved.

## Dataset and preprocessing

The experiments use `fetch_olivetti_faces` from scikit-learn. The dataset has shape `(400, 4096)` and the target vector has shape `(400,)`. The 40 target classes each contain 10 images. The main notebook loads the saved indices from `person_data.npz` rather than creating a new split during each run.

For the robustness check, the notebook reshapes each test image to 64 × 64, zeros the central 16 × 16 block, and flattens the image again before prediction. This allows clean and partially obscured inputs to be compared using the same test identities.

## Running the notebooks in Google Colab

1. Upload `Split.ipynb` and `Ans1_3.ipynb` to Google Drive, or open each notebook directly in Google Colab.
2. Upload `person_data.npz` to the Colab session when the notebook prompts you for it. If the code reads from Google Drive instead, mount Drive and place the file at the path expected by that cell.
3. Open `Split.ipynb` in Colab if you want to inspect the dataset and the saved split.
4. Open `Ans1_3.ipynb` and run its cells from top to bottom using **Runtime → Run all**.
5. If Colab reports a missing package, install it in a code cell with `!pip install package-name`, then rerun the affected cells.

The first dataset fetch may require an internet connection. Colab provides a hosted Python runtime, so a local Jupyter installation is not required. Some experiments—especially repeated resampling and 100-tree comparisons—can take a substantial amount of time. Runtime files can be lost when the session disconnects, so keep the `.npz` file and any results you want to retain in Google Drive or download them before the session ends.

## What the experiments investigate

### 1. Decision-tree baseline

The notebook evaluates combinations of maximum depth (`2`, `4`, `8`, `16`, and unrestricted depth) and minimum leaf size (`1`, `2`, and `4`). Validation accuracy is used to select the tree configuration. In the saved run, the selected tree reached **68.75% validation accuracy**, with **55.00% clean test accuracy**.

### 2. Bagging from bootstrap samples

The bagging experiment trains up to 200 decision trees. Each tree receives a bootstrap sample of 240 training selections drawn with replacement. The notebook evaluates different ensemble sizes and tracks out-of-bag (OOB) accuracy, disagreement between trees, and vote margin.

For the selected 200-tree ensemble, the saved output reports **90.00% validation accuracy**, **90.00% clean test accuracy**, and **77.50% OOB accuracy**. The mean number of distinct training images in a bootstrap sample is close to the theoretical expectation of about 151.9 out of 240.

### 3. Multiclass SAMME boosting

The boosting implementation tests tree depths of 1–4 and requested round counts of 10, 25, 50, and 100. It records validation accuracy, training duration, and learner/weight diagnostics, then compares boosting with bagging at multiple ensemble sizes.

In the final saved comparison, the selected boosting model achieved **35.00% clean test accuracy** and **37.50% masked test accuracy**. These figures describe this particular experiment and configuration, not a general guarantee about boosting.

### 4. Bias–variance and resampling

The notebook repeats training on 30 resampled training sets and compares prediction behaviour for the three approaches. In the saved run, the reported estimates were:

| Model | Mean test accuracy across resamples | Accuracy standard deviation | Bias² estimate | Variance estimate |
|---|---:|---:|---:|---:|
| Single decision tree | 47.13% | 5.77% | 0.3988 | 0.6587 |
| Bagging | 80.33% | 4.14% | 0.5073 | 0.0218 |
| SAMME boosting | 30.17% | 6.32% | 0.8855 | 0.0190 |

These are the values printed by the notebook's implementation of the bias–variance calculation. They should be interpreted under that implementation's definitions and experimental setup.

## Final clean-versus-masked test results

The final comparison cell reports the following accuracies:

| Model | Clean test accuracy | Masked test accuracy |
|---|---:|---:|
| Selected single tree | 55.00% | 2.50% |
| Final bagging ensemble | 90.00% | 73.75% |
| Selected SAMME boosting | 35.00% | 37.50% |

In this run, bagging performs best on both clean and masked test images. The single tree is especially sensitive to the central mask, while the selected boosting configuration has lower accuracy on the clean set. Results may vary if the saved split, random seeds, library versions, or notebook code are changed.

## Outputs

When the relevant cells are executed, the notebook saves tables and plots under `que1_results/`, `que2_results/`, and `que3_results/`. These cover tree-depth tuning, bagging diagnostics, boosting comparisons, and bias–variance analysis. Run the notebook cells that create these folders before expecting the output files to exist.

