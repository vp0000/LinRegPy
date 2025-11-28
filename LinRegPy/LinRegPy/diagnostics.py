import numpy as np
from scipy.stats import chi2, t

class ModelDiagnostics:

    @staticmethod
    def _add_intercept_column(X):
        """Adds a column of ones to X to create the design matrix."""
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    @staticmethod
    def VIF(X, incp=None):
        """Calculate Variance Inflation Factor (VIF) for each feature in X."""
        X = np.asarray(X, float)
        n, p = X.shape
        vifs = np.zeros(p)

        for i in range(p):
            feat = X[:, i]
            others = np.delete(X, i, axis=1)

            # Add intercept to the auxiliary model by default
            X_aux = np.hstack([np.ones((n, 1)), others])

            # Regression: feature_i ~ others
            coef = np.linalg.pinv(X_aux.T @ X_aux) @ (X_aux.T @ feat)
            pred = X_aux @ coef

            SSR = np.sum((pred - np.mean(feat))**2)
            SST = np.sum((feat - np.mean(feat))**2)
            R2 = SSR / SST if SST > 0 else 0.0

            vifs[i] = 1.0 / (1 - R2 + 1e-12)
        if incp is not None: vifs = np.concatenate(([0.0], vifs))
        return vifs
    
    @staticmethod
    def Breusch_Pagan(X, resids, incp=None):
        n, p = X.shape
        e2 = resids**2
        if incp is not None: 
            X_aux = np.hstack([np.ones((n, 1)), X])  # includes intercept
        else: X_aux = X

        coef = np.linalg.pinv(X_aux.T @ X_aux) @ (X_aux.T @ e2)
        pred = X_aux @ coef

        SSR = np.sum((pred - np.mean(e2))**2)
        SST = np.sum((e2 - np.mean(e2))**2)
        R2 = SSR / SST if SST > 0 else 0.0

        test_stat = n * R2
        df = p  # number of regressors (exclude intercept)
        p_val = 1 - chi2.cdf(test_stat, df)

        return {'BP_stat': test_stat, 'BP_p-value': p_val}
    
    @staticmethod
    def Jarque_Bera(resids):
        n = len(resids)
        m = np.mean(resids)
        s2 = np.mean((resids - m)**2)

        skew = np.mean((resids - m)**3) / (s2**1.5 + 1e-12)
        kurt = np.mean((resids - m)**4) / (s2**2 + 1e-12)

        JB = (n/6) * (skew**2 + (kurt - 3)**2 / 4)
        p_val = 1 - chi2.cdf(JB, df=2)

        return {'JB_stat': JB, 'JB_p-value': p_val}
    
    @staticmethod
    # Hypothesis Testing on coefficients for X as used(with/without intercept)
    def hypothesis_testing(X, resids, final_w, confidence_level=0.95):
        n, p = X.shape
        dof = n - p

        # Mean Squared Error
        mse = np.sum(resids**2) / dof

        # Variance-Covariance Matrix
        XtX_inv = np.linalg.pinv(X.T @ X)
        var_cov = mse * XtX_inv

        # Standard Errors
        se_full = np.sqrt(np.diag(var_cov))
    
        # 5. Calculate t-stats, p-values, and CIs
        t_stats = np.divide(final_w, se_full, out=np.zeros_like(final_w), where=se_full!=0)
        p_vals = 2 * (1 - t.cdf(np.abs(t_stats), df=dof))

        alpha_crit = (1 - confidence_level) / 2
        t_crit = t.ppf(alpha_crit, df=dof)
        
        ub = final_w + np.abs(t_crit) * se_full
        lb = final_w - np.abs(t_crit) * se_full
        
        return {
            "se_full": se_full,
            "t_stats": t_stats,
            "p_vals": p_vals,
            "lb": lb,
            "ub": ub
        }

    def regression_diagnostics(self, X, resids, coeffs, incp=None, conf_inv=0.95):
        vifs = self.VIF(X, incp)
        BP = self.Breusch_Pagan(X, resids, incp)
        JB = self.Jarque_Bera(resids)
        if incp:
            X = self._add_intercept_column(X)
        ht = self.hypothesis_testing(X, resids, coeffs, conf_inv)
        regression_diag = {
            "Breusch-Pagan": BP,
            "Jarque-Bera": JB
        }
        return regression_diag, ht, vifs
        


