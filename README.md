**Domain:** E-commerce
# Flipkart E-commerce Decision Intelligence

An AI and Data Science capstone project focused on e-commerce customer analytics, churn prediction, customer segmentation, model evaluation, and neural network architecture analysis.

## Project Overview

This project analyses synthetic e-commerce customer data to understand customer behaviour, identify churn patterns, perform customer segmentation, and evaluate machine learning models.

The project is divided into four parts:

- Part 1: Statistical Analysis and Regression
- Part 2: Classification and Model Evaluation
- Part 3: Clustering and Dimensionality Reduction
- Part 4: Model Selection and Neural Network Architecture

## Project Structure

```text
Flipkart-E-commerce-Decision-Intelligence/
│
├── data/
│   └── records.csv
│
├── outputs/
│   ├── part1_results.txt
│   ├── part2_knn_cv.csv
│   ├── part2_results.txt
│   ├── part2_thresholds.csv
│   ├── part3_hierarchical.csv
│   ├── part3_results.txt
│   ├── part3_segments.csv
│   ├── part4_adaboost.csv
│   ├── part4_architecture.csv
│   ├── part4_results.txt
│   ├── part4_splits.csv
│   └── part4_voting.csv
│
├── generate_data.py
├── common.py
├── part1.py
├── part2.py
├── part3.py
├── part4.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/11249a396-svg/Flipkart-E-commerce-Decision-Intelligence.git
```

### 2. Open the project folder

```bash
cd Flipkart-E-commerce-Decision-Intelligence
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the dataset

```bash
python generate_data.py
```

The dataset is saved as `data/records.csv`.

### 5. Run the Python scripts in the following order

```bash
python part1.py
python part2.py
python part3.py
python part4.py
```

The result files are saved in the `outputs/` directory.

## Part 1 Findings

### Statistical Analysis

The dataset contains 800 records, with a mean order value of 8.9382 hundreds of INR and a standard deviation of 9.1723.

The 95% confidence interval is [8.3026, 9.5738].

The z-statistic is 2.8930, which exceeds the critical value of 1.96. Therefore, the null hypothesis is rejected.

A Type I error would mean concluding that the population's mean order value differs from 8.0 hundreds of INR when, in reality, it is equal to 8.0 hundreds of INR.

The difference between the observed mean and 8.0 is 0.9382 hundreds of INR, equivalent to approximately ₹93.82. This difference may be practically meaningful, although its business importance depends on the scale and context of the orders.

### Sample Size Planning

The required sample size is 1,293, while the available dataset contains 800 records.

Therefore, the available dataset is smaller than the required sample size. The planning team should consider collecting additional records to achieve the desired sample size.

### Regression and Gradient Descent

The closed-form regression produced:

- Slope: 0.6701
- Intercept: -3.6598
- R²: 0.5196
- RMSE: 6.3537
- MAE: 3.7636

The gradient descent slope gap is 0.0000. This indicates that gradient descent converged to the same regression line as the closed-form solution after 500 updates.

### Residual Analysis

The raw residual standard deviation ratio is 3.7525, indicating a funnel pattern where residual spread increases with fitted values.

After applying the log-log transformation, the ratio becomes 1.0396. This suggests that the log transformation largely removes the funnel pattern and produces a more uniform residual spread.

### Multiple Regression

The predictors with absolute coefficients greater than 0.05 are:

- ln(X4): 1.1576
- X2: 0.0940
- X3: 0.1067
- X6: 0.1518

The multiple regression adjusted R² is 0.7403, compared with 0.6561 for the log-log model.

The higher adjusted R² suggests that the additional predictors improve the model's explanatory performance.

## Part 2 Findings

### Data Split and Logistic Regression

The dataset was divided using stratified sampling:

- Training records: 560
- Training positives: 119
- Testing records: 240
- Testing positives: 51

The logistic regression training loss decreased from 0.693147 at epoch 0 to 0.370361 at epoch 1000.

The test binary cross-entropy is 0.381783.

### Learning-Rate Comparison

| Learning rate | Loss after 200 updates |
|---|---:|
| 0.01 | 0.509122 |
| 0.1 | 0.377586 |
| 1.0 | 0.370343 |

The learning rate of 1.0 achieved the lowest loss, followed by 0.1 and 0.01. A learning rate that is too small may converge slowly, while a larger learning rate can reach a lower loss faster when the updates remain stable.

### Important Predictors and Odds Ratios

The three predictors with the largest absolute logistic regression weights are:

- X3: 0.901338
- X1: -0.555353
- X2: 0.550472

X3 has the largest odds ratio of 2.462896. Holding other predictors constant, a one-training-standard-deviation increase in X3 multiplies the estimated odds of churn by approximately 2.46.

### Threshold Evaluation

| Metric | Threshold 0.5 | Threshold 0.3 |
|---|---:|---:|
| Accuracy | 0.804167 | 0.762500 |
| Precision | 0.545455 | 0.463415 |
| Recall | 0.470588 | 0.745098 |
| F1-score | 0.505263 | 0.571429 |

The blind accuracy, obtained by predicting zero for every test record, is 0.787500.

Although the accuracy at threshold 0.5 is 0.804167, it is only slightly higher than blind accuracy. Accuracy alone is therefore not a reliable measure for this imbalanced classification problem.

Lowering the threshold from 0.5 to 0.3 increases recall from 0.470588 to 0.745098, while precision decreases from 0.545455 to 0.463415.

### Logistic Regression versus KNN

| Model | K | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | — | 0.463415 | 0.745098 | 0.571429 |
| KNN | 5 | 0.414634 | 0.333333 | 0.369565 |

Logistic regression performs better than KNN in both recall and F1-score.

### Cost-Based Threshold Selection

The selected threshold using training data is 0.15.

| Evaluation | Cost |
|---|---:|
| Test cost at selected threshold 0.15 | ₹6,050 |
| Test cost at threshold 0.5 | ₹11,800 |
| Test cost when nobody is flagged | ₹20,400 |

Logistic regression with threshold 0.15 is selected for deployment because it produces the lowest reported test cost among these comparisons.

The threshold was selected using training data rather than test data to avoid using test information during model selection and to preserve an honest evaluation.

## Part 3 Findings

### K-Means Clustering and Elbow Method

The WCSS values for different cluster counts are:

| K | WCSS |
|---|---:|
| 2 | 2262.675274 |
| 3 | 1602.847054 |
| 4 | 1494.773166 |
| 5 | 1323.715686 |
| 6 | 1278.169089 |

The successive WCSS drops are:

- WCSS(2) − WCSS(3) = 659.8282
- WCSS(3) − WCSS(4) = 108.0739
- WCSS(4) − WCSS(5) = 171.0575
- WCSS(5) − WCSS(6) = 45.5466

The selected elbow is K = 3.

The WCSS reductions do not shrink steadily because the pinned starting centroids provide a reproducible solution, but not necessarily the global optimum for every K.

### Customer Segment Analysis

The three customer segments are:

| Segment | Size | Mean X1 | Mean X2 | Mean X3 | Mean X4 | Mean X5 | Outcome rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 322 | 47.50 | 1.95 | 0.225 | 12.53 | 62.16 | 1.86% |
| B | 255 | 28.61 | 6.07 | 0.453 | 28.72 | 55.01 | 23.53% |
| C | 223 | 14.39 | 4.23 | 0.712 | 16.51 | 38.52 | 46.64% |

**Segment A – Loyal and Low-Risk Customers**

These customers have longer tenure, fewer complaints and higher loyalty scores. Their outcome rate is only 1.86%.

Business action: Maintain loyalty through personalised benefits and rewards.

**Segment B – High-Value Customers Needing Attention**

These customers have the highest average order value and complaint count.

Business action: Improve complaint resolution and provide better customer support.

**Segment C – High-Risk Customers**

These customers have shorter tenure, higher discount-order share and lower loyalty scores.

Business action: Provide targeted retention offers and improve customer engagement.

Segment C has the highest outcome rate at 46.64%, making it the highest-priority segment for retention activities.

### Hierarchical Clustering

The number of clusters at the selected cut is:

| Linkage | Clusters |
|---|---:|
| Single | 4 |
| Complete | 2 |
| Average | 2 |

The clusters at the cut are:

**Single Linkage:**

- E1
- E2+E3+E6
- E4
- E5

**Complete Linkage:**

- E1+E4+E5
- E2+E3+E6

Single linkage uses the minimum distance between members of two clusters, allowing clusters to join through nearby points and creating a chaining effect. Complete linkage uses the maximum pairwise distance, favouring more compact clusters; therefore, it produces a different grouping at the selected cut.

### Principal Component Analysis (PCA)

The first three principal components explain:

| Component | Explained variance | Cumulative variance |
|---|---:|---:|
| PC1 | 52.27% | 52.27% |
| PC2 | 23.66% | 75.93% |
| PC3 | 10.23% | 86.16% |

Three components are required to explain at least 80% of the variance.

The standardised data has a mean vector of approximately zero.

The PC1 loadings are:

| Predictor | PC1 loading |
|---|---:|
| X1 | 0.531675 |
| X2 | -0.409990 |
| X3 | -0.518712 |
| X4 | -0.291555 |
| X5 | 0.441773 |

PC1 is positively associated with X1 and X5, while X2, X3 and X4 have negative loadings.

The mean PC1 scores are:

- Segment A: 1.777330
- Segment B: -0.783107
- Segment C: -1.670887

Segment A has the highest average PC1 score, while Segment C has the lowest.

## Part 4 Findings

### Decision-Tree Split Selection

The root entropy is 0.746234, and the root Gini impurity is 0.334688.

The best split by both information gain and Gini gain is S4.

| Split | Information gain | Gini gain |
|---|---:|---:|
| S1 | 0.011038 | 0.005178 |
| S2 | 0.003277 | 0.001517 |
| S3 | 0.104596 | 0.049447 |
| S4 | 0.125206 | 0.060644 |

S4 is the best split because it produces the highest information gain and Gini gain among the four candidates.

A tree grown until every leaf is pure can overfit noise and small patterns in the training data. A maximum-depth limit is one rule that can prevent a tree from growing too deep.

### Model Comparison and Voting

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Logistic | 0.4634 | 0.7451 | 0.5714 |
| KNN | 0.4146 | 0.3333 | 0.3696 |
| Stump | 0.4333 | 0.7647 | 0.5532 |
| Hard voting | 0.4375 | 0.6863 | 0.5344 |
| Soft voting | 0.4368 | 0.7451 | 0.5507 |

Logistic regression is the preferred model because it has the highest F1-score of 0.5714.

Combining models did not improve the F1-score. The decision stump achieved the highest recall of 0.7647, while logistic regression achieved a recall of 0.7451.

### AdaBoost Analysis

The first weak learner has an error of 0.25 and an alpha value of 0.549306.

The two misclassified points after round 1 are points 3 and 7. Each receives a normalised weight of 0.25, so their combined weight is 0.50.

The second weak learner has an error of 0.166667 and an alpha value of 0.804719. Since its alpha is higher than the first learner's alpha, the second weak learner receives greater influence in the final ensemble.

The final ensemble accuracy is 0.75.

### Neural Network Architecture

**Model A – CNN Architecture (Option 2)**

Input: 28 × 28 × 1 grayscale image.

| Layer | Output shape | Parameters |
|---|---|---:|
| Input | 28 × 28 × 1 | 0 |
| Convolution 1 + ReLU | 28 × 28 × 6 | 156 |
| Max Pool 1 | 14 × 14 × 6 | 0 |
| Convolution 2 + ReLU | 10 × 10 × 16 | 2,416 |
| Max Pool 2 | 5 × 5 × 16 | 0 |
| Flatten | 400 | 0 |
| Dense 1 + ReLU | 32 | 12,832 |
| Output Dense | 10 | 330 |

Total CNN parameters: **15,734**

The MLP architecture has 13,002 parameters.

For a 10-class classification task, ReLU is suitable for hidden layers, while softmax is suitable for the output layer with categorical cross-entropy loss.

**Fully Connected versus Convolutional Layer**

| Layer type | Parameters |
|---|---:|
| Fully connected layer | 50,332,672 |
| Convolutional layer | 1,792 |

Convolutional layers use shared filters, which greatly reduces the number of parameters compared with fully connected layers.

**Model B – RNN Architecture**

| Matrix | Shape | Parameters |
|---|---|---:|
| W_xh | 128 × 64 | 8,192 |
| W_hh | 128 × 128 | 16,384 |
| W_hy | 16 × 128 | 2,048 |

Bias parameters:

- Hidden bias: 128
- Output bias: 16

Total RNN parameters: **26,768**

The RNN parameter count does not increase with sequence length because the same weight matrices are reused at every time step.

In very deep plain networks, gradients can become extremely small during backpropagation, preventing early layers from learning effectively. Skip connections provide a direct path for information and gradients, helping deeper networks train more effectively.

## Model-Choice Memo

**To: Head of the Retention Team**

**Subject: Recommendation for E-commerce Customer Retention**

I recommend deploying the logistic regression model as the initial structured-data solution for customer churn prediction.

The model achieved an F1-score of 0.5714 and recall of 0.7451 at a threshold of 0.3. The KNN model achieved a lower F1-score of 0.3696 and recall of 0.3333. The voting models also did not improve the logistic model's F1-score.

For operational deployment, I recommend the threshold of 0.15 selected using training data. Its test cost is ₹6,050, compared with ₹11,800 at threshold 0.5 and ₹20,400 when nobody is flagged.

The main limitation is that the analysis uses a synthetic dataset and a single fixed train-test split, so real-world performance may differ.

Model A, the CNN for product images, and Model B, the RNN for customer-review text, are not immediate priorities for churn prediction; they can be explored as separate future projects.

**Recommendation:** Start with logistic regression at threshold 0.15 and validate it using real customer data before production deployment.

## Limitations

1. The project uses a synthetic dataset, so its patterns may not represent actual customer behaviour.
2. Model evaluation uses one fixed stratified train-test split rather than repeated validation across multiple splits.
3. The test set contains only 51 positive records, which limits the certainty of performance estimates.
4. The retention cost assumptions of ₹50 per false alarm and ₹400 per missed positive are simplified business estimates.
5. The neural network architectures are analysed through parameter counts and shapes; they are not trained or evaluated on actual image or text datasets.
6. Customer segmentation and PCA results depend on the selected features, scaling method and initial clustering choices.

## Conclusion

This capstone project demonstrates the application of statistical analysis, regression, classification, clustering, dimensionality reduction, ensemble learning and neural network architecture analysis to an e-commerce customer analytics problem.

The results identify customer segments with different risk levels and show that logistic regression provides the strongest F1-score among the evaluated classification models.

The project recommends using logistic regression as an initial churn prediction model, with further validation on real-world customer data before production deployment.

---

**Project:** Flipkart E-commerce Decision Intelligence  
**Domain:** E-commerce  
**Program:** Minor Degree in Artificial Intelligence & Data Science  
**Institution:** IIT Mandi
