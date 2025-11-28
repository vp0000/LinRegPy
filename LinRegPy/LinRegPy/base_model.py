import numpy as np
import pandas as pd
from typing import Dict
from .optimizer import GradientDescent
from .diagnostics import ModelDiagnostics
from datetime import datetime
import pickle as pkl

class LinearRegSuper:
    """
    A unified class for Linear Regression supporting OLS, Ridge, Lasso, and Huber 
    Loss via Gradient Descent or the Normal Equation.
    """
    def __init__(self, method_dict: Dict, fit_intercept: bool=True, w_init=None, feature_names=None):
        """
        Initializes the regressor with method parameters and intercept settings.

        :param method_dict: Dictionary containing the 'name' (ols, ridge, etc.) and hyperparameters.
        :param fit_intercept: If True, adds a bias term (intercept).
        :param w_init: Optional initial weight vector for Gradient Descent.
        """
        self.fit_intercept = fit_intercept
        self.method_dict = method_dict.copy()
        self.w_init = w_init
        self.optimizer = GradientDescent() # Initialize the separate optimizer class
        self.incp = 0.0                    # Fitted intercept
        self.coeff = np.array([])          # Fitted coefficients (excluding intercept)
        self.cost_hist = []                # Cost history from Gradient Descent
        self.scale = False                 # Scale flag for indicating if dataset is normalized or not
        self.means = None
        self.stds = None
        self._fitted = False
        self._grad_desc = False
        self.feature_names = None
    
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        incp_flag = self.fit_intercept

        # Add intercept column if required
        if incp_flag:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        n, m = X.shape  # n = samples, m = params (intercept + features)
        method_params = self.method_dict.copy()
        method_name = method_params.get('name', 'ols')

        # determine whether to use GD (None -> default for lasso/huber)
        use_gd = method_params.get('use_gd', None)
        if use_gd is None:
            use_gd = method_name in ['lasso', 'huber']

        # Decide scaling behaviour.
        # Default: scale for lasso/ridge/huber OR whenever use_gd is True (except explicit OLS scalings).
        # If you want explicit control, accept method_params['scale'].
        scale_flag = method_params.get('scale', None)
        if scale_flag is None:
            scale_flag = (method_name in ['lasso', 'ridge', 'huber']) or (use_gd and method_name != 'ols')
        self.scale = bool(scale_flag)

        # Compute feature means/stds and replace features with scaled ones if scaling enabled
        if self.scale:
            X_feats = X[:, 1:] if incp_flag else X
            self.means = np.mean(X_feats, axis=0)
            self.stds = np.std(X_feats, axis=0) + 1e-12
            X_scaled_feats = (X_feats - self.means) / self.stds
            if incp_flag:
                X[:, 1:] = X_scaled_feats
            else:
                X = X_scaled_feats

        # Build initial w_0 (in internal scaled coordinate system if scaling used)
        if self.w_init is not None:
            # if scale was applied, convert user-supplied w_init (assumed in original units)
            w_0 = self.w_init.copy()
            if self.scale:
                if incp_flag:
                    w_0[1:] = w_0[1:] * self.stds  # beta_scaled = beta_raw * stds
                else:
                    w_0 = w_0 * self.stds
            # else w_0 already matches internal unscaled X
        else:
            w_0 = np.zeros(m)

        # --- Solver selection ---
        if use_gd:
            self._grad_desc = True
            grad_loss_dict = {
                'ols': ['ols_grad', 'ols_cost'],
                'ridge': ['ridge_grad', 'ridge_cost'],
                'lasso': ['lasso_grad', 'lasso_cost'],
                'huber': ['huber_grad', 'huber_cost']
            }
            grad_name, cost_name = grad_loss_dict.get(method_name, ['ols_grad', 'ols_cost'])

            lr = method_params.get('lr', 0.01)
            n_iter = method_params.get('n_iter', 1000)
            mult = method_params.get('mult', 0.0)  # alpha or delta

            # Dynamic function lookup
            grad_func = getattr(self.optimizer, grad_name)
            cost_func = getattr(self.optimizer, cost_name)

            # Run gradient descent (you confirmed base_gradient_descent exists)
            w, self.cost_hist = self.optimizer.base_gradient_descent(
                X, y, w_0, lr, n_iter, grad_func, mult, cost_func, incp_flag
            )
        else:
            self._grad_desc = False
            reg_mult = method_params.get('mult', 0.0)

            # Build reg matrix carefully. Two choices:
            # A) reg in scaled space (alpha applies after scaling): reg = alpha * I
            # B) reg in original units (alpha interpreted on original X): reg diagonal scaled by std^2
            # Here I implement B (alpha in original units) since that's usually what users expect.
            if method_name == 'ridge':
                reg = np.zeros((m, m))

                if incp_flag:
                    # intercept not regularized
                    reg[0, 0] = 0.0
                    # feature diagonals: alpha * std_j^2 (map original-scale penalty into scaled space)
                    if self.scale:
                        assert self.stds is not None
                        # self.stds is length p (features only)
                        reg[1:, 1:] = np.diag(reg_mult * (self.stds**2))
                    else:
                        reg[1:, 1:] = np.diag(np.repeat(reg_mult, m - 1))
                else:
                    if self.scale:
                        assert self.stds is not None
                        reg[:] = np.diag(np.concatenate(([0.0], reg_mult * (self.stds**2))))[:m, :m]
                    else:
                        reg = np.identity(m) * reg_mult
            else:
                reg = np.zeros((m, m))

            # Normal equation solution
            inv_term = X.T @ X + reg

            # Check exact singularity (rank of X including intercept)
            if np.allclose(reg, np.zeros((m, m))) and np.linalg.matrix_rank(X) < m:
                raise ValueError("Perfect multicollinearity present, OLS not possible")

            inv = np.linalg.pinv(inv_term)
            w = inv @ (X.T @ y)
            self.cost_hist = []

        # --- Post-Fitting: Separate Intercept and Coefficients ---
        if incp_flag:
            w0 = w[0]
            w_feat = w[1:]
        else:
            w0 = 0.0
            w_feat = w

        # Unscale coefficients back to original units (if scaling applied)
        if self.scale:
            assert self.stds is not None
            assert self.means is not None
            coeff_unscaled = w_feat / self.stds
            incp_unscaled = w0 - np.sum((w_feat * self.means) / self.stds)
            self.incp = float(incp_unscaled)
            self.coeff = coeff_unscaled.copy()
        else:
            self.incp = float(w0)
            self.coeff = w_feat.copy()

        self._fitted = True
        return self

    def predict(self, X):
        """
        Generates predictions for new data X. 
        X must NOT contain the intercept column.
        """
        X = np.asarray(X, dtype=float)
        if self._fitted == False:
            raise ValueError("Error: Model not fitted, please use fit() before predict()")
        return np.dot(X, self.coeff) + self.incp

    def score(self, X, y):
        """
        Calculates the R-squared and Adjusted R-squared scores.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        y_hat = self.predict(X)
        
        # Calculate Sum of Squares
        SSE = np.sum((y - y_hat)**2)
        SST = np.sum((y - np.mean(y))**2)
        n = len(y)
        k = len(self.coeff) # Number of independent predictors (features)
        
        if SST == 0:
            return {'R^2': 1.0 if SSE == 0 else 0.0, 'Adj. R^2': np.nan}
            
        r_square = 1 - SSE/SST
        
        # Adjusted R^2 Calculation (Check for degrees of freedom)
        if n - k - 1 <= 0:
             adj_r_square = np.nan
        else:
             adj_r_square = 1 - (1 - r_square) * ((n - 1) / (n - k - 1))
             
        return {'R^2': r_square, 'Adj. R^2': adj_r_square}

    def evaluate(self, X, y):
        """
        Calculates common regression error metrics: MSE, MAE, and RMSE.
        """
        X, y, y_hat = self._get_predictions_and_arrays(X, y)
        
        error = y - y_hat
        
        mse = np.mean(error**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(error))
        
        return {'MSE': mse, 'MAE': mae, 'RMSE': rmse}

    def summary_report(self, X, y, feature_names=None, float_fmt="{:0.4f}", perc_fmt="{:0.2%}"):
        """
        Generates a detailed regression summary report including coefficients,
        standard errors, t-stats, p-values, and overall model metrics.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape  # n = samples, p = features (excluding intercept)

        if self._fitted == False:
            raise ValueError("Error: Model not fitted, please use fit() before summary_report()")

        # Prepare feature names
        if feature_names is None:
            feature_names = [f"X{i+1}" for i in range(p)]
        if self.fit_intercept:
            feature_names = ['Intercept'] + feature_names

        resids = y - self.predict(X)
        coeffs = np.concatenate(([self.incp], self.coeff)) if self.fit_intercept else self.coeff
        incp = self.incp if self.fit_intercept else None
        # Compute standard errors, t-stats, p-values
        diagnostics = ModelDiagnostics()
        model_diag, model_ht, model_vifs = diagnostics.regression_diagnostics(X, resids, coeffs, incp=incp, conf_inv=0.95)
        score_metrics = self.score(X, y)
        eval_metrics = self.evaluate(X, y)

        # header info
        hdr_lines = []
        hdr_lines.append("OLS Regression Results" if self.method_dict.get("name","ols")=="ols" else f"{self.method_dict.get('name','regression').upper()} Regression Results")
        hdr_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        hdr_lines.append(f"Method: {self.method_dict.get('name','ols')}")
        hdr_lines.append(f"Scale applied: {bool(self.scale)}")
        hdr_lines.append(f"Gradient Descent used: {bool(self._grad_desc)}")
        hdr_lines.append(f"Number of obs: {X.shape[0]}")
        hdr_lines.append(f"Degrees of Freedom: {n - (p + 1) if self.fit_intercept else n - p}")
        
        #Diagnostics
        diag_lines = []
        for key, value in score_metrics.items():
            diag_lines.append(f"{key}: {perc_fmt.format(value)}")
        for key, value in eval_metrics.items():
            diag_lines.append(f"{key}: {float_fmt.format(value)}")
        for key, value in model_diag.items():
            diag_lines.append(f"{key}":)
            for k, v in value.items():
                diag_lines.append(f"{k.split('_')[1]}: {float_fmt.format(v)}")

        # Coefficient DataFrame
        coeff_data = {
            "Coefficient": coeffs,
        }
        coeff_data.update(model_ht)  # Add Std. Error, t-Statistic, p-Value, Lower/Upper Bound
        coeff_data["VIF"] = model_vifs
        coeff_data = pd.DataFrame(coeff_data).set_index(pd.Index(feature_names))
        coeff_data.index.name = "Variable"
        coeff_cols = coeff_data.columns.tolist()
        new_cols = [
            'Coefficient',
            'Std. Error',
            't-Statistic',
            'p-Value',
            'Lower Bound',
            'Upper Bound',
            'VIF'
        ]
        coeff_data.rename(columns=dict(zip(coeff_cols, new_cols)), inplace=True)

        #Coeffiecient table
        table_lines = []
        header = ["Variable", "Coef", "Std.Err", "t", "P>|t|", "[0.025", "0.975]", "VIF"]
        table_lines.append("{:20s} {:>12s} {:>12s} {:>10s} {:>10s} {:>8s} {:>8s} {:>8s}".format(*header))
        table_lines.append("-" * 100)

        # for non-OLS, mask inference columns to NaN (to avoid misinterpretation)
        is_ols = (self.method_dict.get("name","ols") == "ols")

        for name, row in coeff_data.iterrows():
            coef = row["Coefficient"]
            se = row.get("Std. Error", np.nan)
            tstat = row.get("t-Statistic", np.nan) if is_ols else np.nan
            pval = row.get("p-Value", np.nan) if is_ols else np.nan
            ub = row.get("Upper Bound", np.nan) if is_ols else np.nan
            lb = row.get("Lower Bound", np.nan) if is_ols else np.nan
            vif = row.get("VIF", np.nan)

            coef_s = float_fmt.format(coef) if np.isfinite(coef) else "nan"
            se_s = float_fmt.format(se) if np.isfinite(se) else "nan"
            t_s = float_fmt.format(tstat) if np.isfinite(tstat) else "nan"
            p_s = perc_fmt.format(pval) if np.isfinite(pval) else "nan"
            ub_s = float_fmt.format(ub) if np.isfinite(tstat) else "nan"
            lb_s = float_fmt.format(lb) if np.isfinite(tstat) else "nan"
            vif_s = float_fmt.format(vif) if np.isfinite(vif) else "nan"

            table_lines.append("{:20s} {:>12s} {:>12s} {:>10s} {:>10s} {:>8s} {:>8s} {:>8s}".format(
                str(name), coef_s, se_s, t_s, p_s, lb_s, ub_s, vif_s
            ))
        
        # Combine all parts into final report
        report_lines = []
        report_lines.append("\n".join(hdr_lines))
        report_lines.append("\nCoefficient Estimates:")
        report_lines.append("\n".join(table_lines))
        report_lines.append("\nModel Diagnostics:")
        report_lines.append("\n".join(diag_lines))
        report_lines.append("- Note: For non-OLS methods, inference statistics may not be valid.")
        report_lines.append("- VIF > 5 indicates moderate multicollinearity; >10 indicates high multicollinearity.")
        report_lines.append("- BP p-value < 0.05 indicates heteroscedasticity.")
        report_lines.append("- JB p-value < 0.05 indicates non-normal residuals.")
        print("\n".join(report_lines))
        return report_lines

    def get_final_params(self):
        """Returns the fitted intercept and feature coefficients."""
        return {'Intercept': self.incp, 'Coefficients': self.coeff}
    
    def get_cost_history(self):
        """Returns the list of cost values tracked during Gradient Descent."""
        return self.cost_hist
    
    def save_model(self, filepath: str):
        """
        Saves the current LinearRegSuper model instance to the specified file path using pickle.

        :param filepath: Path where the model will be saved.
        """
        try:
            with open(filepath, 'wb') as f:
                pkl.dump(self, f)
            print(f"Model saved at path: {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    @staticmethod
    def load(filepath: str):
        """
        Loads a saved LinearRegSuper model from the specified file path.

        :param filepath: Path to the file containing the saved model.
        :return: Loaded LinearRegSuper model instance or None if loading fails.
        """
        try:
            with open(filepath, 'rb') as f:
                model = pkl.load(f)
            print("Model successfully loaded")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    # --- HELPER METHODS ADDED FOR EVALUATION FUNCTIONS ---
    def _get_predictions_and_arrays(self, X, y):
        """Internal helper to standardize array conversion and prediction."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        y_hat = self.predict(X)
        return X, y, y_hat
    
    def _get_feature_names(self, X, feature_names=None):
        if feature_names is not None:
            return feature_names
        # default generic names

        return [f"x{i+1}" for i in range(X.shape[1])]






