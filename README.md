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
```

    ## Loading required package: ggplot2

    ## Loading required package: StanHeaders

    ## rstan (Version 2.12.1, packaged: 2016-09-11 13:07:50 UTC, GitRev: 85f7a56811da)

    ## For execution on a local, multicore CPU with excess RAM we recommend calling
    ## rstan_options(auto_write = TRUE)
    ## options(mc.cores = parallel::detectCores())

``` r
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
N = 500
x1 = rnorm(N)
x2 = rnorm(N)
X = cbind(1, x1, x2)
beta = c(-1,.2,-.3)
mu = plogis(X%*%beta)  # add noise if desired + rnorm(N, sd=.01)
phi = 10
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
brmod <- betareg(y ~ ., data = data.frame(y, X[,-1]))
summary(brmod)
```

    ## 
    ## Call:
    ## betareg(formula = y ~ ., data = data.frame(y, X[, -1]))
    ## 
    ## Standardized weighted residuals 2:
    ##     Min      1Q  Median      3Q     Max 
    ## -3.4724 -0.5896  0.0495  0.6394  2.7698 
    ## 
    ## Coefficients (mean model with logit link):
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -1.00491    0.03135 -32.051  < 2e-16 ***
    ## x1           0.16651    0.03082   5.403 6.57e-08 ***
    ## x2          -0.28723    0.03064  -9.373  < 2e-16 ***
    ## 
    ## Phi coefficients (precision model with identity link):
    ##       Estimate Std. Error z value Pr(>|z|)    
    ## (phi)   9.6437     0.5904   16.34   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 
    ## 
    ## Type of estimator: ML (maximum likelihood)
    ## Log-likelihood: 322.3 on 4 Df
    ## Pseudo R-squared: 0.1805
    ## Number of iterations: 16 (BFGS) + 2 (Fisher scoring)

Stan
----

We can fit the same model Stan with the following code:

``` r
stan_beta <- "
data {
  int<lower=1> N;
  int<lower=1> K;
  vector<lower=0,upper=1>[N] y;
  matrix[N,K] X;

}

parameters {
  vector[K] gamma;
  real<lower=0> phi;
}
transformed parameters{
  vector<lower=0,upper=1>[N] mu;    // transformed linear predictor
  vector<lower=0>[N] A;             // parameter for beta distn
  vector<lower=0>[N] B;             // parameter for beta distn

  for (i in 1:N)
    mu[i] = inv_logit(X[i,] * gamma);   
  A = mu * phi;
  B = (1.0 - mu) * phi;
}
model {

  // priors
  // gamma ~ normal(0, 1);   
  // phi ~ cauchy(0, 5);               // different options for phi  
  // phi ~ inv_gamma(.001, .001);
  // phi ~ uniform(0, 500);          // put upper on phi if using this
  // likelihood
  y ~ beta(A, B);

}
"
# Stan data list
dat = list(N = length(y), K=ncol(X), y=y, X=X)

beta_stan_test <- stan(model_code = stan_beta,
     data       = dat,
     pars       = c("gamma", "phi"))
```

    ## 
    ## SAMPLING FOR MODEL '382b1c68da14e9cc8bda0f2a11f59ecc' NOW (CHAIN 1).
    ## 
    ## Chain 1, Iteration:    1 / 2000 [  0%]  (Warmup)
    ## Chain 1, Iteration:  200 / 2000 [ 10%]  (Warmup)
    ## Chain 1, Iteration:  400 / 2000 [ 20%]  (Warmup)
    ## Chain 1, Iteration:  600 / 2000 [ 30%]  (Warmup)
    ## Chain 1, Iteration:  800 / 2000 [ 40%]  (Warmup)
    ## Chain 1, Iteration: 1000 / 2000 [ 50%]  (Warmup)
    ## Chain 1, Iteration: 1001 / 2000 [ 50%]  (Sampling)
    ## Chain 1, Iteration: 1200 / 2000 [ 60%]  (Sampling)
    ## Chain 1, Iteration: 1400 / 2000 [ 70%]  (Sampling)
    ## Chain 1, Iteration: 1600 / 2000 [ 80%]  (Sampling)
    ## Chain 1, Iteration: 1800 / 2000 [ 90%]  (Sampling)
    ## Chain 1, Iteration: 2000 / 2000 [100%]  (Sampling)
    ##  Elapsed Time: 2.121 seconds (Warm-up)
    ##                2.258 seconds (Sampling)
    ##                4.379 seconds (Total)
    ## 
    ## [1] "The following numerical problems occured the indicated number of times after warmup on chain 1"
    ##                                                                                          count
    ## Exception thrown at line 32: beta_log: Second shape parameter[2] is 0, but must be > 0!      4
    ## Exception thrown at line 32: beta_log: Second shape parameter[65] is 0, but must be > 0!     1
    ## [1] "When a numerical problem occurs, the Hamiltonian proposal gets rejected."
    ## [1] "See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected"
    ## [1] "If the number in the 'count' column is small,  do not ask about this message on stan-users."
    ## 
    ## SAMPLING FOR MODEL '382b1c68da14e9cc8bda0f2a11f59ecc' NOW (CHAIN 2).
    ## 
    ## Chain 2, Iteration:    1 / 2000 [  0%]  (Warmup)
    ## Chain 2, Iteration:  200 / 2000 [ 10%]  (Warmup)
    ## Chain 2, Iteration:  400 / 2000 [ 20%]  (Warmup)
    ## Chain 2, Iteration:  600 / 2000 [ 30%]  (Warmup)
    ## Chain 2, Iteration:  800 / 2000 [ 40%]  (Warmup)
    ## Chain 2, Iteration: 1000 / 2000 [ 50%]  (Warmup)
    ## Chain 2, Iteration: 1001 / 2000 [ 50%]  (Sampling)
    ## Chain 2, Iteration: 1200 / 2000 [ 60%]  (Sampling)
    ## Chain 2, Iteration: 1400 / 2000 [ 70%]  (Sampling)
    ## Chain 2, Iteration: 1600 / 2000 [ 80%]  (Sampling)
    ## Chain 2, Iteration: 1800 / 2000 [ 90%]  (Sampling)
    ## Chain 2, Iteration: 2000 / 2000 [100%]  (Sampling)
    ##  Elapsed Time: 2.191 seconds (Warm-up)
    ##                1.976 seconds (Sampling)
    ##                4.167 seconds (Total)
    ## 
    ## [1] "The following numerical problems occured the indicated number of times after warmup on chain 2"
    ##                                                                                          count
    ## Exception thrown at line 32: beta_log: Second shape parameter[6] is 0, but must be > 0!      3
    ## Exception thrown at line 32: beta_log: Second shape parameter[44] is 0, but must be > 0!     1
    ## [1] "When a numerical problem occurs, the Hamiltonian proposal gets rejected."
    ## [1] "See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected"
    ## [1] "If the number in the 'count' column is small,  do not ask about this message on stan-users."
    ## 
    ## SAMPLING FOR MODEL '382b1c68da14e9cc8bda0f2a11f59ecc' NOW (CHAIN 3).
    ## 
    ## Chain 3, Iteration:    1 / 2000 [  0%]  (Warmup)
    ## Chain 3, Iteration:  200 / 2000 [ 10%]  (Warmup)
    ## Chain 3, Iteration:  400 / 2000 [ 20%]  (Warmup)
    ## Chain 3, Iteration:  600 / 2000 [ 30%]  (Warmup)
    ## Chain 3, Iteration:  800 / 2000 [ 40%]  (Warmup)
    ## Chain 3, Iteration: 1000 / 2000 [ 50%]  (Warmup)
    ## Chain 3, Iteration: 1001 / 2000 [ 50%]  (Sampling)
    ## Chain 3, Iteration: 1200 / 2000 [ 60%]  (Sampling)
    ## Chain 3, Iteration: 1400 / 2000 [ 70%]  (Sampling)
    ## Chain 3, Iteration: 1600 / 2000 [ 80%]  (Sampling)
    ## Chain 3, Iteration: 1800 / 2000 [ 90%]  (Sampling)
    ## Chain 3, Iteration: 2000 / 2000 [100%]  (Sampling)
    ##  Elapsed Time: 2.199 seconds (Warm-up)
    ##                2.116 seconds (Sampling)
    ##                4.315 seconds (Total)
    ## 
    ## [1] "The following numerical problems occured the indicated number of times after warmup on chain 3"
    ##                                                                                           count
    ## Exception thrown at line 32: beta_log: Second shape parameter[3] is 0, but must be > 0!       3
    ## Exception thrown at line 32: beta_log: First shape parameter[313] is 0, but must be > 0!      1
    ## Exception thrown at line 32: beta_log: Second shape parameter[2] is 0, but must be > 0!       1
    ## Exception thrown at line 32: beta_log: Second shape parameter[313] is 0, but must be > 0!     1
    ## [1] "When a numerical problem occurs, the Hamiltonian proposal gets rejected."
    ## [1] "See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected"
    ## [1] "If the number in the 'count' column is small,  do not ask about this message on stan-users."
    ## 
    ## SAMPLING FOR MODEL '382b1c68da14e9cc8bda0f2a11f59ecc' NOW (CHAIN 4).
    ## 
    ## Chain 4, Iteration:    1 / 2000 [  0%]  (Warmup)
    ## Chain 4, Iteration:  200 / 2000 [ 10%]  (Warmup)
    ## Chain 4, Iteration:  400 / 2000 [ 20%]  (Warmup)
    ## Chain 4, Iteration:  600 / 2000 [ 30%]  (Warmup)
    ## Chain 4, Iteration:  800 / 2000 [ 40%]  (Warmup)
    ## Chain 4, Iteration: 1000 / 2000 [ 50%]  (Warmup)
    ## Chain 4, Iteration: 1001 / 2000 [ 50%]  (Sampling)
    ## Chain 4, Iteration: 1200 / 2000 [ 60%]  (Sampling)
    ## Chain 4, Iteration: 1400 / 2000 [ 70%]  (Sampling)
    ## Chain 4, Iteration: 1600 / 2000 [ 80%]  (Sampling)
    ## Chain 4, Iteration: 1800 / 2000 [ 90%]  (Sampling)
    ## Chain 4, Iteration: 2000 / 2000 [100%]  (Sampling)
    ##  Elapsed Time: 2.164 seconds (Warm-up)
    ##                2.025 seconds (Sampling)
    ##                4.189 seconds (Total)
    ## 
    ## [1] "The following numerical problems occured the indicated number of times after warmup on chain 4"
    ##                                                                                           count
    ## Exception thrown at line 32: beta_log: Second shape parameter[2] is 0, but must be > 0!       4
    ## Exception thrown at line 32: beta_log: Second shape parameter[166] is 0, but must be > 0!     1
    ## [1] "When a numerical problem occurs, the Hamiltonian proposal gets rejected."
    ## [1] "See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected"
    ## [1] "If the number in the 'count' column is small,  do not ask about this message on stan-users."

Which yields the following output:

``` r
summary(beta_stan_test)$summary
```

    ##                 mean      se_mean         sd        2.5%         25%
    ## gamma[1]  -1.0057790 0.0004782705 0.03024848  -1.0642012  -1.0272746
    ## gamma[2]   0.1664441 0.0004603292 0.02911377   0.1096610   0.1471425
    ## gamma[3]  -0.2870644 0.0004942303 0.02993236  -0.3449547  -0.3074917
    ## phi        9.6375716 0.0102017748 0.56065998   8.5417864   9.2717795
    ## lp__     322.7166774 0.0291576921 1.31505108 319.3909697 322.0763129
    ##                  50%         75%       97.5%    n_eff      Rhat
    ## gamma[1]  -1.0062704  -0.9860921  -0.9456055 4000.000 1.0008322
    ## gamma[2]   0.1669939   0.1863161   0.2231476 4000.000 1.0003099
    ## gamma[3]  -0.2876310  -0.2664294  -0.2278951 3667.948 0.9996441
    ## phi        9.6336181   9.9952216  10.7587095 3020.283 1.0003403
    ## lp__     323.0359997 323.6735115 324.3143122 2034.131 1.0010029
