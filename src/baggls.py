import numpy as np
from scipy.stats import norm, truncnorm, invgamma
from scipy.linalg import lapack

class BaGGLS:
    """
    Bayesian group global-local shrinkage (BaGGLS) Model

    This class implements a Bayesian model for probit regression subject to
    our novel group global-local shrinkage prior using variational Bayes 
    with coordinate ascent optimization. The model is built upon a design matrix derived from 
    motif features and their interactions, and it fits a binary response.

    Attributes:
        y (np.ndarray): Response vector.
        n (int): Number of observations.
        d (int): Number of motifs (features).
        design_matrix (np.ndarray): Predictor matrix including intercept, motif-specific effects, and interactions.
        indicator (np.ndarray): Matrix indicating which motifs each effect is based on.
        p (int): Total number of effects (columns in the design matrix).
        effect_names (list): List of names corresponding to each effect for easier interpretation.
    """

    def __init__(self, response, motifs, binary=False, standardize=True, cut_quantile=0.05):
        """
        Initialize the BGGLS model with response data and motif features.

        Parameters:
            response (np.ndarray): The binary response vector with size n.
            motifs (pd.DataFrame): DataFrame containing non-negative attribution scores for d motifs over n observations.
            binary (bool): If True, use a binary indicator for motif activity; otherwise, use the actual motif scores (default: False).
            standardize (bool): If True, standardize columns of the design matrix to have scale 1 (default: True).
            cut_quantile (float): Quantile threshold for filtering effects based on contribution scores (default: 0.05).
        """
        super().__init__()
        self.y = response   # Store the response vector.
        self.n, self.d = motifs.shape  # n observations and d motifs.
        self.design_matrix, self.indicator = self.build_design_matrix(motifs, binary=binary, standardize=standardize, cut_quantile=cut_quantile)
        self.p = len(self.design_matrix[0])  # Number of effects (columns in the design matrix).
        
        
    def build_design_matrix(self, motifs, binary=False, standardize=True, cut_quantile=0.05):
        """
        Construct the design matrix and corresponding indicator matrix based on motif data.
    
        Parameters:
            motifs (pd.DataFrame): DataFrame of motif attribution scores.
            binary (bool): If True, binarize motif activity; otherwise, use normalized scores (default: False).
            standardize (bool): If True, standardize columns of the design matrix to have scale 1 (default: True).
            cut_quantile (float): Quantile threshold for filtering effects based on contribution scores (default: 0.05).
            
        Returns:
            tuple:
                design_matrix (np.ndarray): The final predictor matrix with shape (n, p).
                indicator (np.ndarray): A matrix of shape (p, d) that indicates which motifs contribute to each effect.
        """
        n, d = motifs.shape
        
        # Normalize attribution scores to sum to 1 per row
        motifs = motifs.div(motifs.sum(axis=1),axis=0)
        
        indicator = []
        design_matrix = []
        self.effect_names = []
        
        # Add intercept term
        design_matrix.append(np.ones(n))
        self.effect_names.append('intercept')
        indicator.append(np.zeros(d))
        
        # Loop over each motif to add individual and interaction effects
        for s1 in range(d):
            indicator_local = np.zeros(d)
            indicator_local[s1] = 1
            indicator.append(indicator_local)
            design_matrix.append(motifs.iloc[:, s1])
            self.effect_names.append(motifs.columns[s1])
            for s2 in range(s1 + 1, d):
                indicator_local = np.zeros(d)
                indicator_local[[s1, s2]] = 1
                indicator.append(indicator_local)
                self.effect_names.append(motifs.columns[s1] + ' AND ' + motifs.columns[s2])
                design_matrix.append(motifs.iloc[:, s1]*motifs.iloc[:, s2])
            
        # Convert lists to NumPy arrays and adjust the shapes
        design_matrix = np.array(design_matrix, dtype=np.float32)
        design_matrix = np.vstack(design_matrix).T
        indicator = np.vstack(indicator)
        self.effect_names = np.asarray(self.effect_names)
        
        # Compute contribution scores for each effect and filter based on quantile threshold
        contributions_effects = design_matrix.sum(axis=0)/n
        thresh = np.quantile(contributions_effects, q=1-cut_quantile)
        
        design_matrix = design_matrix[:,contributions_effects>=thresh]
        indicator = indicator[contributions_effects>=thresh,:]
        self.effect_names = self.effect_names[contributions_effects>=thresh]
        
        p = len(design_matrix[0])
        
        if binary:
            design_matrix = 1*(design_matrix>0)
        
        if standardize:
            for l in range(1,p):
                if np.sum(design_matrix[:,l]>0) > 1 and len(np.unique(design_matrix[:,l]))>2:
                    design_matrix[:,l] = design_matrix[:,l]/np.std(design_matrix[:,l][design_matrix[:,l]>0])
        
        return design_matrix, indicator

    def fit_vb(self, max_num_steps=10000, threshold=0.001, verbose=False):
        """
        Fit the BGGLS model using Variational Bayes with coordinate ascent optimization.

        This method initializes variational parameters and then iteratively updates them until convergence 
        is achieved (i.e., the change in the beta mean is below the specified threshold) or until the maximum 
        number of iterations is reached.

        Parameters:
            max_num_steps (int): Maximum number of iterations for the optimization (default: 10000).
            threshold (float): Convergence threshold based on the change in beta estimates (default: 0.001).
            verbose (bool): If True, print convergence information at each iteration (default: False).

        Updates various variational parameters such as sigma_beta, b_beta, mu_z, sigma2_z, a_tau, b_tau, 
        a_nu, b_nu, a_lambda, b_lambda, a_c, b_c, a_delta, b_delta, a_t, and b_t.
        """
        # Initialize variational parameters.
        self.sigma_beta = np.identity(self.p)
        self.b_beta = np.matmul(self.sigma_beta, self.design_matrix.T)
        self.mu_z = np.zeros(self.n)
        self.sigma2_z = np.ones(self.n)
        self.a_tau = (self.p + 1) / 2
        self.b_tau = 1 / 2
        self.a_nu = 1
        self.b_nu = 1 / 2
        self.a_lambda = np.ones(self.p)
        self.b_lambda = np.ones(self.p)
        self.a_c = np.ones(self.p)
        self.b_c = 0.5 * np.ones(self.p)
        self.a_delta = (np.sum(self.indicator, axis=0) + 1) / 2
        self.b_delta = 1 / 2 * np.ones(self.d)
        self.a_t = np.ones(self.d)
        self.b_t = 1 / 2 * np.ones(self.d)
        
        step = 0
        beta_mean, beta_mean_old = np.ones(self.p), np.zeros(self.p)
        while (np.linalg.norm(beta_mean - beta_mean_old) > threshold and step < max_num_steps):
            # Update sigma_beta using an efficient inversion method.
            self.sigma_beta = self.efficently_inverse_xtxplusd(
                self.design_matrix,
                self.a_tau / self.b_tau * self.a_lambda / self.b_lambda * np.matmul(self.indicator, self.a_delta / self.b_delta)
            )
            self.b_beta = np.matmul(self.sigma_beta, self.design_matrix.T)
            
            # Compute the expected value of latent variable z.
            exp_z = self.mu_z + np.sqrt(self.sigma2_z) * (2 * self.y - 1) * norm.pdf(self.mu_z / np.sqrt(self.sigma2_z)) / norm.cdf((2 * self.y - 1) * self.mu_z / np.sqrt(self.sigma2_z))
            
            matrix_for_mu_z = np.matmul(self.sigma_beta,np.matmul(self.design_matrix.T, exp_z))
            for i in range(self.n):
                self.mu_z[i] = self.sigma2_z[i] * np.matmul(
                    self.design_matrix[i, :],matrix_for_mu_z-np.matmul(self.sigma_beta,exp_z[i]*self.design_matrix[i, :].T)
                )  
            
            self.sigma2_z = 1 / (1 - np.diag(np.matmul(self.design_matrix, np.matmul(self.sigma_beta, self.design_matrix.T))))
            exp_z = self.mu_z + np.sqrt(self.sigma2_z) * (2 * self.y - 1) * norm.pdf(self.mu_z / np.sqrt(self.sigma2_z)) / norm.cdf((2 * self.y - 1) * self.mu_z / np.sqrt(self.sigma2_z))
            exp_beta2 = np.diag(self.sigma_beta + np.matmul(self.sigma_beta, 
                             np.matmul(self.design_matrix.T, 
                             np.matmul(np.matmul(np.diag(self.sigma2_z - (exp_z - self.mu_z) * exp_z), 
                             self.design_matrix), self.sigma_beta)))) + np.matmul(self.b_beta, exp_z) ** 2
            
            self.b_tau = 0.5 * np.sum(self.a_lambda / self.b_lambda * exp_beta2 * np.matmul(self.indicator, self.a_delta / self.b_delta)) + self.a_nu / self.b_nu
            self.b_nu = self.a_tau / self.b_tau
            self.b_lambda = self.a_tau / self.b_tau * 0.5 * exp_beta2 * np.matmul(self.indicator, self.a_delta / self.b_delta) + self.a_c / self.b_c
            self.b_c = 1 + self.a_lambda / self.b_lambda
            self.b_delta = self.a_tau / self.b_tau * np.matmul(self.indicator.T, self.a_lambda / self.b_lambda * exp_beta2) + self.a_t / self.b_t
            self.b_t = 1 + self.a_delta / self.b_delta

            beta_mean_old, beta_mean = beta_mean, np.matmul(np.matmul(self.sigma_beta, self.design_matrix.T), exp_z)
            step += 1
            if verbose:
                print(f"Step {step}: convergence = {np.linalg.norm(beta_mean - beta_mean_old):.6f}")

    def efficently_inverse_xtxplusd(self, x, d_diag):
        """
        Efficiently compute the inverse of (X^T * X + D), where D is a diagonal matrix.

        This function uses two approaches:
          - If the number of parameters (p) is greater than the number of observations (n), it uses the Woodbury matrix identity 
            combined with Cholesky decomposition for efficient inversion.
          - If p <= n, it performs a direct inversion using the Cholesky decomposition.

        Parameters:
            x (np.ndarray): Design matrix of shape (n, p).
            d_diag (np.ndarray): 1D array containing the diagonal elements of matrix D.

        Returns:
            np.ndarray: The inverse of (X^T * X + D).
        """
        d_diag = np.clip(d_diag, 1e-300, None)
        n = len(x)
        p = len(x[0])
        if p > n:
            # Use Woodbury matrix identity.
            d_inv = np.diag(1 / d_diag)
            capacitance = np.identity(n) + np.matmul(np.matmul(x, d_inv), x.T)
            cholesky = lapack.dpotrf(capacitance)[0]
            cholesky_inv = lapack.dpotri(cholesky)[0]
            capacitance_inv = cholesky_inv + np.triu(cholesky_inv, k=1).T
            return d_inv - np.matmul(np.matmul(np.matmul(d_inv, x.T), capacitance_inv), np.matmul(x, d_inv))
        else:
            # Direct inversion using Cholesky decomposition.
            matrix = np.matmul(x.T, x) + np.diag(d_diag)
            cholesky = lapack.dpotrf(matrix)[0]
            cholesky_inv = lapack.dpotri(cholesky)[0]
            return cholesky_inv + np.triu(cholesky_inv, k=1).T

    def sample_from_q_beta(self, num_samples):
        """
        Sample from the variational distribution q(beta).

        This method generates samples from the posterior distribution of beta using the current variational parameters.
        It first samples from a multivariate normal distribution and then adjusts the samples using a truncated normal 
        component.

        Parameters:
            num_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: A 2D array of shape (num_samples, p) containing samples from the distribution q(beta).
        """
        samples = []
        u0 = np.random.multivariate_normal(np.matmul(self.b_beta, self.mu_z), self.sigma_beta, size=num_samples)
        for i in range(num_samples):
            # Sample from a truncated normal distribution for adjustment.
            u1 = truncnorm.rvs(a=-(2 * self.y - 1) * self.mu_z / np.sqrt(self.sigma2_z), b=np.inf)
            samples.append(u0[i] + np.matmul(self.b_beta, (2 * self.y - 1) * np.sqrt(self.sigma2_z) * u1))
        return np.vstack(samples)
    
    def summarize_effects(self, alpha=0.95, num_samples=50000):
        """
        Summarize the model effects using credible intervals for variable selection.

        This method computes the posterior mean for beta and uses samples from q(beta) to determine which effects 
        are statistically significant. If the alpha-credible interval for an effect includes zero, the effect is set to zero.
        Otherwise, the effect's name and estimated value are printed.

        Parameters:
            alpha (float): Credible interval level (default: 0.95).
            num_samples (int): Number of samples to draw from the posterior distribution for computing credible intervals (default: 50000).

        Returns:
            np.ndarray: The posterior mean estimates of beta after performing variable selection.
        """
        exp_z = self.mu_z + np.sqrt(self.sigma2_z) * (2 * self.y - 1) * norm.pdf(self.mu_z / np.sqrt(self.sigma2_z)) / norm.cdf((2 * self.y - 1) * self.mu_z / np.sqrt(self.sigma2_z))
        beta_mean = np.matmul(self.b_beta, exp_z)
        beta_samples = self.sample_from_q_beta(num_samples=num_samples)
        for j in range(self.p):
            lower_bound = np.quantile(beta_samples[:, j], q=(1 - alpha) / 2)
            upper_bound = np.quantile(beta_samples[:, j], q=1 - ((1 - alpha) / 2))
            # If the credible interval for beta[j] includes 0, set its estimate to 0.
            if lower_bound < 0 and upper_bound > 0:
                beta_mean[j] = 0 
            else:
                print(self.effect_names[j], np.round(beta_mean[j], 3))
        return beta_mean

    @staticmethod
    def nan_overlapping(motif1, motif2):
        """
        Calculate the percentage of overlapping non-NaN values between two motif arrays.

        This helper method computes the proportion of samples for which both input motifs have valid (non-NaN) values.

        Parameters:
            motif1 (np.ndarray or pd.Series): First motif's values.
            motif2 (np.ndarray or pd.Series): Second motif's values.

        Returns:
            float: The percentage of samples where both motif1 and motif2 have non-NaN values.
        """
        non_nan_overlap = (~np.isnan(motif1)) & (~np.isnan(motif2))
        overlap_count = np.sum(non_nan_overlap)
        total_samples = len(motif1)
        percentage_overlap = (overlap_count / total_samples) * 100
        return percentage_overlap

    def absolute_effects(self, top_n=20, verbose=False):
        """
        Calculate the absolute effects of the model.

        This method computes the absolute effects of the model and returns the top N effects.

        Parameters:
            top_n (int): Number of top effects to return (default: 20).
            verbose (bool): Whether to print the effects (default: False).
        """
        exp_z = self.mu_z + np.sqrt(self.sigma2_z) * (2 * self.y - 1) * norm.pdf(self.mu_z / np.sqrt(self.sigma2_z)) / norm.cdf((2 * self.y - 1) * self.mu_z / np.sqrt(self.sigma2_z))
        beta_mean = np.matmul(self.b_beta, exp_z)
        abs_effects = np.abs(beta_mean)
        top_indices = np.argsort(abs_effects)[-top_n:]
        top_indices = top_indices[::-1]
        top_effect_names = [self.effect_names[idx] for idx in top_indices]
        top_effect_values = [beta_mean[idx] for idx in top_indices]
        if verbose:
            print(f"Top {top_n} strongest effects:")
            for idx in top_indices:
                print(f"{self.effect_names[idx]}: {beta_mean[idx]:.6f}")
        return top_effect_names, top_effect_values