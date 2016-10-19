Beta Regression in Stan
================

Beta Regression in Stan
-----------------------

This is code to implement beta regression similar to the [`betareg` package](https://cran.r-project.org/web/packages/betareg/index.html) in [Stan](http://mc-stan.org/).

Simulated data
--------------

A minimal simulation model can be used to relate the value of a beta distributed random variable, `y`, to two covariates, `x1` and `x2`.

``` r
library(betareg)
library(rstan)
library(dplyr)


N = 500
x1 = rnorm(N)
x2 = rnorm(N)
X = cbind(1, x1, x2)
beta = c(-1,.2,-.3)
gamma = c(2, -.25, .5)
mu = plogis(X %*% beta) 
phi = exp(X %*% gamma)
A = mu*phi
B = (1-mu)*phi
y = rbeta(N, A, B)

qplot(y, geom = "histogram", binwidth = .025)
```

![](README_files/figure-markdown_github/ggplot2-1.png)

betareg
-------

The simulated data can be fit using the `betareg` package as follow:

``` r
brmod <- betareg(y ~ x1 + x2 | x1 + x2, data = data.frame(y, X[,-1]))
summary(brmod)
```

    ## 
    ## Call:
    ## betareg(formula = y ~ x1 + x2 | x1 + x2, data = data.frame(y, X[, 
    ##     -1]))
    ## 
    ## Standardized weighted residuals 2:
    ##     Min      1Q  Median      3Q     Max 
    ## -3.4843 -0.6696  0.0722  0.7284  2.6681 
    ## 
    ## Coefficients (mean model with logit link):
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -1.04739    0.03494 -29.978  < 2e-16 ***
    ## x1           0.24659    0.03227   7.641 2.16e-14 ***
    ## x2          -0.32298    0.03360  -9.612  < 2e-16 ***
    ## 
    ## Phi coefficients (precision model with log link):
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  2.06417    0.06035  34.201  < 2e-16 ***
    ## x1          -0.29964    0.05826  -5.143 2.71e-07 ***
    ## x2           0.44351    0.05951   7.453 9.14e-14 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 
    ## 
    ## Type of estimator: ML (maximum likelihood)
    ## Log-likelihood: 303.6 on 6 Df
    ## Pseudo R-squared: 0.1502
    ## Number of iterations: 17 (BFGS) + 2 (Fisher scoring)

Stan
----

We can fit the same model Stan with the following code:

``` r
stan_beta <- "
data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  vector<lower=0,upper=1>[N] y;
  matrix[N,K] X;
  matrix[N,J] Z;
}

parameters {
  vector[K] beta;
  vector[J] gamma;
}

transformed parameters{
  vector<lower=0,upper=1>[N] mu;    // transformed linear predictor for mean of beta distribution
  vector<lower=0>[N] phi;           // transformed linear predictor for precision of beta distribution
  vector<lower=0>[N] A;             // parameter for beta distn
  vector<lower=0>[N] B;             // parameter for beta distn

  for (i in 1:N) {
    mu[i]  = inv_logit(X[i,] * beta);   
    phi[i] = exp(Z[i,] * gamma);
  }

  A = mu .* phi;
  B = (1.0 - mu) .* phi;
}

model {
  // priors

  // likelihood
  y ~ beta(A, B);
}

generated quantities{
  vector[N] log_lik;
  vector[N] log_lik_rep;
  vector<lower=0,upper=1>[N] y_rep;
  real total_log_lik;
  real total_log_lik_rep;
  
  int<lower=0, upper=1> p_omni;

  for (n in 1:N) {
    log_lik[n] = beta_lpdf(y[n] | A[n], B[n]);
    y_rep[n] = beta_rng(A[n], B[n]);
    log_lik_rep[n] = beta_lpdf(y_rep[n] | A[n], B[n]);
  }

  total_log_lik = sum(log_lik);
  total_log_lik_rep = sum(log_lik_rep);

  p_omni = (total_log_lik_rep > total_log_lik);
}
"

# Stan data list
dat = list(N = length(y), 
            K = dim(X)[2], 
            J = dim(X)[2], 
            y = y, 
            X = X, 
            Z = X)

beta_stan_test <- stan(model_code = stan_beta,
     data       = dat,
     pars       = c("beta", "gamma"))
```

Which yields the following output. Note that I'm currently having trouble with initial values causing the chain the fail to return samples. If you try to run this code, you may get the same error. If you try again, you may hit upon passable initial values which give you usable output.

``` r
summary(beta_stan_test)$summary
```

    ##                 mean      se_mean         sd        2.5%         25%
    ## beta[1]   -1.0460158 0.0007021041 0.03457468  -1.1141740  -1.0683737
    ## beta[2]    0.2465325 0.0006105098 0.03148848   0.1852846   0.2252508
    ## beta[3]   -0.3224054 0.0006281683 0.03275012  -0.3877300  -0.3443261
    ## gamma[1]   2.0547838 0.0011911642 0.05959231   1.9390481   2.0136568
    ## gamma[2]  -0.2970869 0.0010678717 0.05455385  -0.4066153  -0.3345093
    ## gamma[3]   0.4436024 0.0010202432 0.05500592   0.3334926   0.4074231
    ## lp__     300.6970820 0.0379956499 1.71476123 296.4338028 299.8250876
    ##                  50%         75%       97.5%    n_eff      Rhat
    ## beta[1]   -1.0462534  -1.0231162  -0.9783571 2425.008 0.9995212
    ## beta[2]    0.2467158   0.2674112   0.3098302 2660.223 1.0017947
    ## beta[3]   -0.3226130  -0.3006249  -0.2583619 2718.152 1.0007457
    ## gamma[1]   2.0551450   2.0956233   2.1744043 2502.863 1.0011350
    ## gamma[2]  -0.2954093  -0.2604913  -0.1925056 2609.832 0.9998897
    ## gamma[3]   0.4444233   0.4803254   0.5508495 2906.775 0.9996765
    ## lp__     301.0345727 301.9528785 302.9774667 2036.759 1.0024270
