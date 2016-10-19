library(betareg)
library(rstan)
library(dplyr)

s = seq(0, 1, 0.01)
mu = .9
phi = 5
a = mu*phi
b = (1-mu)*phi
qplot(s, dbeta(s, a,b), geom = "line")



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
hist(y, 'FD')



# model for later comparison
brmod <- betareg(y ~ x1 + x2 | x1 + x2, data = data.frame(y, X[,-1]))
summary(brmod)



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
                       pars       = c("beta", "gamma"),
                       chains = 5)



summary(beta_stan_test)$summary
summary(brmod)
