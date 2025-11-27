# LinRegPy

This project is a hobby venture and demonstration of core statistical learning principles, implementing a unified estimator for various linear regression models entirely from scratch using Python and NumPy, with pandas being used for summary creation.

It focuses on **transparency, modularity, and comprehensive diagnostics** as a pedagogical exercise aimed at understanding under the hood processes in advanced libraries.

### Key Features and Design Focus

| Feature | Engineering Principle |
| :--- | :--- |
| **Unified Estimator** | Single `LinearRegSuper` class handles **OLS**, **Ridge**, **Lasso**, and **Huber** regression. |
| **Custom Optimizer** | **Gradient Descent** algorithm is implemented generically for now, accepting any gradient function (`Lasso`, `Huber`) as a modular input. |
| **Dual Solvers** | Automatically switches between **Normal Equation** (for speed on OLS/Ridge) and **Gradient Descent** (for non-closed form solutions like Lasso/Huber), depending on the user's input |
| **Statistical Diagnostics** | Generates a full report including **Hypothesis Testing** ($t$-stats, $p$-values), **VIF** (Multicollinearity), **Breusch-Pagan** (Heteroscedasticity), and **Jarque-Bera** (Residual Normality). |
| **Correct Inference Handling** | Automatically masks $t$-stats, $p$-values, and Confidence Intervals for **Ridge/Lasso/Huber** models, acknowledging that these are invalid for biased estimators. |

---

### Data Preparation and Internal Scaling

This library requires that all **feature engineering** be done manually before calling `.fit()`, but it handles **Standard Scaling** internally to ensure reliable convergence for iterative methods.

The library automatically performs internal Standard Scaling on the feature matrix $\mathbf{X}$ when:
1. The chosen `method_name` in the method dictionary is **`lasso`**, **`ridge`**, or **`huber`**.
2. The solver is set to **Gradient Descent (`use_gd=True`)**.

Crucially, the final coefficients and intercept are **unscaled** back to the original units before storing them, ensuring the final summary report displays interpretable values.

---

### Quick Start and API Usage

#### 1. Installation

Clone the repository and install the required dependencies in development mode:

```bash
git clone [https://github.com/YourUsername/PyStatLin.git](https://github.com/YourUsername/PyStatLin.git)
cd PyStatLin
pip install numpy pandas scipy # Install requirements
pip install -e .              # Install in editable/development mode
```
#### 2. Sample Usage

The run_example.py file has a baseline implementation of the OLS method on the California Housing Dataset at scikit-learn(https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html). You can play around with the method_dict and analyse the results for multiple models at once, based on your requirements. 

---

### Limitations and Scope

This is a hobby project aimed at understanding the backend processes used by ML libraries and scikit-learn and statsmodels, and is not intended for production as of now.
Other limitations include:
1. Inability to choose between methods for gradient descent and diagnostic tests based on user input.
2. Inability to visualise and analyse the model through graphing.
3. Lack of influence analysis and ANOVA.
4. Lack of hyperparameter tuning and cross-validation, requiring the user to supply the optimal parameters by themselves.
5. Performance challenges on large datasets due to the Pure-Python structure.
These will be rectified in future versions with proper documentation and implementation, with possible releases in the pipeline as the library is enhanced through updates.

#### Disclosure

I have used LLMs for assistance in debugging, minor refactoring and text generation.

---

Thank you for going through my project. I am available at varun3122000@gmail.com for contact regarding suggestions, comments and any other communication as necessary.

