library(betareg)
library(rstan)
library(dplyr)

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
hist(y, 'FD')

# model for later comparison
brmod <- betareg(y ~ ., data = data.frame(y, X[,-1]))
summary(brmod)


# Stan data list
dat = list(N = length(y), K=ncol(X), y=y, X=X)

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


beta_stan_test <- stan(model_code = stan_beta,
     data       = dat,
     pars       = c("gamma", "phi"))

summary(beta_stan_test)$summary
summary(brmod)
