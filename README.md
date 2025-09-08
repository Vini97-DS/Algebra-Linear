# Machine Learning for Insurance: Customer Analysis and Data Protection

### Project Overview

This project was undertaken for the "Protect Your Tomorrow" insurance company to explore the application of machine learning in solving key business tasks. The primary objectives were to leverage ML for customer analysis and prediction, and to develop a robust data transformation algorithm to protect clients' personal information without compromising model quality.

The project addresses four main challenges:

* Customer Segmentation: Identifying similar clients to aid marketing efforts.

* Payout Prediction (Classification): Predicting whether a client is likely to receive an insurance payout.

* Benefit Prediction (Regression): Predicting the number of insurance benefits a client is likely to receive.

* Data Obfuscation: Anonymizing sensitive client data while maintaining the accuracy of the regression model.

### Methodology and Tasks

**Task 1: Finding Similar Customers (k-NN)**
To help marketing agents identify similar clients, a k-Nearest Neighbors (k-NN) algorithm was implemented. The analysis focused on the impact of feature scaling and distance metrics.

  * Algorithms: k-Nearest Neighbors (k-NN).

  * Metrics: Euclidean and Manhattan distances.

  * Key Finding: The analysis demonstrated that unscaled data significantly skews the results. Features with larger numeric scales (like income) dominate the distance calculation, making      them disproportionately important. After applying MaxAbsScaler, all features contributed equally, leading to a more accurate and meaningful identification of similar clients.

**Task 2: Payout Prediction (Classification)**
This task was framed as a binary classification problem to predict whether a client would receive an insurance benefit.

  * Models:

    * A k-NN classifier was built and evaluated for k=1 through 10.

    * A dummy model based on random probability was used as a baseline.

  * Metric: F1-Score.

  * Results: The k-NN classifier, especially on scaled data, vastly outperformed the baseline. The best k-NN model achieved an F1-Score of 0.945, compared to the dummy model's best score       of only 0.20. This confirms the value of the ML approach over random chance.

**Task 3: Benefit Prediction (Regressinon)**
A Linear Regression model was built from scratch using NumPy to predict the number of insurance benefits a new client would receive.

  * Model: Custom Linear Regression class using the analytical solution w=(XT X)−1 XTy.

  * Metrics: Root Mean Squared Error (RMSE) and R-squared (R²).

  * Results: The model performed consistently on both original and scaled data, achieving an RMSE of 0.34 and an R² of 0.66. This indicated that, for this linear model, feature scaling did not impact the final prediction quality, unlike distance-based algorithms like k-NN.

**Task 4: Data Obfuscation**
The final and most critical task was to develop a method to protect clients' personal data. This was achieved by obfuscating the feature matrix (X) by multiplying it by an invertible random matrix (P).

  * Transformation: X′=X timesP

  * Proof of Concept: It was proven both analytically (mathematically) and computationally (via code) that this transformation does not affect the quality of the Linear Regression model.

    * Analytical Proof: The weights of the transformed model (w_P) are related to the original weights (w) by w_P=P−1w. The final predictions remain identical because haty_P=X ′w_P=(XP)(P−w)=Xw=haty.

    * Computational Proof: The Linear Regression model trained on the obfuscated data produced the exact same RMSE (0.34) and R² (0.66) as the model trained on the original data.

### Conclusions

This project successfully demonstrates the power of machine learning and linear algebra in a practical business context. The key takeaways are:

  1. Feature Scaling is Critical: For distance-based algorithms like k-NN, proper feature scaling is not just a best practice but a necessity for generating meaningful results.

  2. ML Models Provide Significant Value: The k-NN classifier demonstrated a massive improvement in predictive power over a simple baseline model, proving its utility for the business.

  3. Data Privacy Can Be Preserved: The data obfuscation technique is a powerful tool. It allows the company to protect sensitive customer information effectively without sacrificing the        predictive performance of its linear models.

### Technologies Used

  * Python

  * Pandas: For data manipulation and analysis.

  * NumPy: For numerical operations and linear algebra.

  * Scikit-learn: For scaling, k-NN modeling, and performance metrics.

  * Seaborn: For data visualization.

  * Jupyter Notebook: As the development environment.
