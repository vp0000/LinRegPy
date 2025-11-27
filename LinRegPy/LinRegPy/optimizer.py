import numpy as np

class GradientDescent: 
    # Please note that regularizer penalties like L1 and L2 are not scaled in this version
    # --- CORE HELPER METHODS ---
    
    def _calculate_residuals(self, X, y, w):
        """Calculates the error vector (y_hat - y)."""
        return X @ w - y
        
    def _calculate_ols_grad_base(self, X, y, err):
        """Calculates the non-regularized (OLS/MSE) gradient part."""
        n = len(y)
        return (1 / n) * X.T @ err

    def ols_cost(self, X, y, w, alpha=0.0, incp=True):
        """Calculates the standard Mean Squared Error (MSE) Cost."""
        err = self._calculate_residuals(X, y, w)
        return np.mean(err**2) / 2
    
    # --- GRADIENT METHODS (Using Helpers) ---
    
    def ols_grad(self, X, y, w, alpha=0.0, incp=True):
        err = self._calculate_residuals(X, y, w)
        return self._calculate_ols_grad_base(X, y, err)

    def ridge_grad(self, X, y, w, alpha=0.0, incp=True):
        err = self._calculate_residuals(X, y, w)
        ols_grad = self._calculate_ols_grad_base(X, y, err)
        
        # L2 Penalty Gradient
        penalty_grad = 2 * alpha * w
        if incp:
            penalty_grad[0] = 0.0
            
        return ols_grad + penalty_grad

    def lasso_grad(self, X, y, w, alpha=0.0, incp=True):
        err = self._calculate_residuals(X, y, w)
        ols_grad = self._calculate_ols_grad_base(X, y, err)
        
        # L1 Penalty Subgradient
        lasso_grad = alpha * np.sign(w)
        if incp:
            lasso_grad[0] = 0.0
            
        return ols_grad + lasso_grad
    
    # Note: huber_grad doesn't reuse ols_grad_base.
    def huber_grad(self, X, y, w, delta=1.0, incp=True):
        err = self._calculate_residuals(X, y, w)
        g = np.where(np.abs(err) <= delta, err, delta * np.sign(err))
        grad = (1 / len(y)) * X.T @ g
        return grad

    # --- COST METHODS (Using Helpers) ---
    
    def ridge_cost(self, X, y, w, alpha=0.0, incp=True):
        mse_cost = self.ols_cost(X, y, w)
        
        # L2 Penalty Term
        w_reg = w[1:] if incp else w # Select weights to penalize
        l2_penalty = alpha * np.sum(w_reg**2)
        
        return mse_cost + l2_penalty
    
    def lasso_cost(self, X, y, w, alpha=0.0, incp=True):
        mse_cost = self.ols_cost(X, y, w)
        
        # L1 Penalty Term
        w_reg = w[1:] if incp else w # Select weights to penalize
        l1_penalty = alpha * np.sum(np.abs(w_reg))
        
        return mse_cost + l1_penalty
        
    def huber_cost(self, X, y, w, delta=1.0, incp=True):
        err = self._calculate_residuals(X, y, w)
        abs_err = np.abs(err)
        
        # Piecewise calculation
        quadratic = 0.5 * err**2
        linear = delta * (abs_err - 0.5 * delta)
        
        cost_vector = np.where(abs_err <= delta, quadratic, linear)
        return np.mean(cost_vector)

    # --- OPTIMIZATION ENGINE (Remains the same) ---
    def base_gradient_descent(
            self, X, y, w_0, lr, n_max, grad_func, mult=0.0, cost_func=None, incp=True
        ):
        w = w_0.copy()
        cost_history = []
        for _ in range(n_max):
            grad = grad_func(X, y, w, mult, incp)
            #grad = np.clip(grad, -1e3, 1e3)
            w = w - lr * grad
            cost = cost_func(X, y, w, mult, incp) if cost_func is not None else np.nan
            if cost_func is not None and not(np.isfinite(cost)): break
            cost_history.append(cost)
            if len(cost_history) >= 2:
                if abs(cost_history[-1] - cost_history[-2]) < 1e-6:
                    if not(np.isnan(cost_history[-1]) or np.isnan(cost_history[-2])):
                        break #Simple early stop for now
        return w, cost_history