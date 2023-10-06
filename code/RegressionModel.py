#!/usr/bin/python
# Programmer : Jiayin Hong
# Created: 21 Feb 2023 11:43:20 AM

#### import dependent modules ####
import numpy as np
from scipy import stats

class RegressionModel:
# Note that this class expects exog with constant!!
# pre-process the regressor matrix as follows:
# exog = np.column_stack([np.ones(X_var.shape[0]), X_var])

    def __init__(self, endog, exog, params):
        self.endog = endog
        self.exog = exog
        self.nobs = float(self.exog.shape[0])
        self._df_model = None
        self.df_resid = None
        self.rank = None
        self._rsquared = None
        self._loglike = None

        self.resid = self.endog - np.dot(self.exog, params)
        # Sum of squared residuals
        self.ssr = np.sum(self.resid**2)

        centered_endog = self.endog - self.endog.mean()

        # The total sum of squares centered about the mean
        self.centered_tss = np.dot(centered_endog, centered_endog)      
        # The explained sum of squares
        self.ess = self.centered_tss - self.ssr

    @property
    def df_model(self):
        """
        The model degree of freedom.
        Defined as the rank of the regressor matrix minus 1 if a
        constant is included.
        """
        if self._df_model is None:
            if self.rank is None:
                self.rank = np.linalg.matrix_rank(self.exog)
            self._df_model = float(self.rank - 1)
        return self._df_model

    @df_model.setter
    def df_model(self, value):
        self._df_model = value

    @property
    def df_resid(self):
        """
        The residual degree of freedom.
        Defined as the number of observations minus the rank of
        the regressor matrix.
        """
        if self._df_resid is None:
            if self.rank is None:
                self.rank = np.linalg.matrix_rank(self.exog)
            self._df_resid = self.nobs - self.rank
        return self._df_resid

    @df_resid.setter
    def df_resid(self, value):
        self._df_resid = value

    def predict(self, params, exog=None):
        """
        Return linear predicted values from a design matrix.

        Parameters
        ----------
        params : array_like
            Parameters of a linear model. Intercept first, then slope.
        exog : array_like, optional
            Design / exogenous data. Model exog is used if None.

        Returns
        -------
        array_like
            An array of fitted values.
        """
        if exog is None:
            exog = self.exog
        return np.dot(exog, params)

    @property
    def loglike(self):
        """
        The likelihood function for the OLS model.

        Parameters
        ----------
        params : array_like
            The coefficients with which to estimate the log-likelihood.

        Returns
        -------
        float
            The profile (concentrated) likelihood function evaluated at params.
        """
        if self._loglike is None:
            nobs2 = self.nobs / 2.0
            nobs = float(self.nobs)
            # profile log likelihhod
            self._loglike = -nobs2*np.log(2*np.pi) - nobs2*np.log(self.ssr / nobs) - nobs2
        return self._loglike

    @loglike.setter
    def loglike(self, value):
        self._loglike = value

    @property
    def rsquared(self):
        """
        R-squared of the model.

        Defined as 1 - `ssr`/`centered_tss` if the constant is
        included in the model.
        """
        if self._rsquared is None:
            self._rsquared = 1 - self.ssr/self.centered_tss
        return self._rsquared

    @rsquared.setter
    def rsquared(self, value):
        self._rsquared = value

    def rsquared_adj(self):
        """
        Adjusted R-squared.

        Defined as 1 - (`nobs`-1)/`df_resid` * (1-`rsquared`)
        if a constant is included.
        """
        return 1 - (np.divide(self.nobs - 1, self.df_resid)
                    * (1 - self.rsquared))

    def f_statistic(self):
        """
        F-statistic of the model & the p-value of the F-statistic.

        Calculated as the mean squared error of the model divided by the mean
        squared error of the residuals if the nonrobust covariance is used.
        """

        # Mean squared error of the model, defined as the explained sum of
        # squares divided by the model degree of freedom.
        mse_model = self.ess/self.df_model

        # Mean squared error of the residuals, defined as the sum of squared
        # residuals divided by the residual degree of freedom.
        mse_resid = self.ssr/self.df_resid

        f_statistic = mse_model/mse_resid
        f_pvalue = stats.f.sf(f_statistic, self.df_model, self.df_resid)
        return f_statistic, f_pvalue

    def info_criteria(self):
        """Return an information criterion for the model.

        Returns
        ----------
        AIC, BIC

        References
        ----------
        Burnham KP, Anderson KR (2002). Model Selection and Multimodel
        Inference; Springer New York.
        """
        # Akaike's information criteria. Calculated as `-2*loglike + 2(df_model + 1)`
        aic = -2*self.loglike + 2*(self.df_model+1)

        # Bayes' information criteria. Calculated as `-2*loglike + log(n_observation)*(df_model+1)`
        bic = -2*self.loglike + np.log(self.nobs)*(self.df_model+1)
        return aic, bic

    def qr_factorization(self):
        Q, R = np.linalg.qr(self.exog)
        effects = np.dot(Q.T, self.endog)
        # estimate of intercept and slope
        beta = np.linalg.solve(R, effects)
        return beta

