#!/usr/bin/python
# Programmer : Jiayin Hong
# Created: 25 Sep 2023 13:30

#### import dependent modules ####
import numpy as np
import statsmodels.api as sm
from scipy.optimize import lsq_linear
from RegressionModel import RegressionModel
from scale_plot_super import scale_plot_super

def statsRunner(datax, datay):
    """
    This function runs optimal non-constrained regression for the sublinear model,
    and constrained regression for the linear model. It returns statistics for the
    use of plotting.
    """
    # optimal non-constrained regression by statsmodels
    X_var = np.log10(datax.astype(float)).values.reshape(-1,1)
    Y_var = np.log10(datay.astype(float)).values.reshape(-1,1)
    X2 = sm.add_constant(X_var) # by default, no intercept is added by the model
    est = sm.OLS(Y_var, X2)
    # record info in a dict to later pass to plot function
    sublinear = dict(Slope=est.fit().params[1], Intercept=est.fit().params[0],
                    Rsquared=est.fit().rsquared, Pvalue=est.fit().f_pvalue,
                    BIC=est.fit().bic)

    # constrained regression by scipy
    A = X2 # regressor matrix
    b = Y_var.flatten() # target vector
    lb = np.array([-np.inf, 1.000]) # lower bounds on intercept, slope
    ub = np.array([np.inf, 1.001]) # upper bounds on intercept, slope
    res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto', verbose=1)
    params = res.x # optimal intercept while slope is constrained
    # get stats for constrained linear-scaling model
    linear_mock = RegressionModel(b, A, params)
    # record info in a dict
    linear = dict(Slope=params[1], Intercept=params[0],
                Rsquared=linear_mock.rsquared, Pvalue=linear_mock.f_statistic()[1],
                BIC=linear_mock.info_criteria()[1])

    return sublinear, linear






