data {
  int<lower=1> N;
  int<lower=1> N_persons;
  int<lower=1> N_items;
  int<lower=1> T_max;
  array[N] int<lower=1,upper=N_persons> person;
  array[N] int<lower=1,upper=N_items>   item;
  array[N] int<lower=1,upper=T_max>     time;
  array[N] int<lower=0,upper=1>         response;
}

parameters {
  // Hold item difficulty constant over time
  vector[N_items] beta;

  // Person ability: linear growth per person
  vector[N_persons] theta_0;
  // Growth rate per person
  vector[N_persons] theta_growth;

  // Noise (SD) around the ability trajectory
  real<lower=0> sigma;

  // Observation-level ability deviation
  matrix[N_persons, T_max] eta;
}

transformed parameters {
  // Ability at each time point for each person
  matrix[N_persons, T_max] theta;
  for (p in 1:N_persons)
    for (t in 1:T_max)
      theta[p, t] = theta_0[p] + theta_growth[p] * t + sigma * eta[p, t];
}

model {
  // Priors
  beta          ~ normal(0, 1);
  theta_0       ~ normal(0, 1);
  theta_growth  ~ normal(0, 0.01);
  sigma         ~ exponential(1);
  to_vector(eta) ~ normal(0, 1);

  // Likelihood using Rasch model
  for (n in 1:N)
    response[n] ~ bernoulli_logit(
      theta[person[n], time[n]] - beta[item[n]]
    );
}

generated quantities {
  // Log-likelihood
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = bernoulli_logit_lpmf(
      response[n] | theta[person[n], time[n]] - beta[item[n]]
    );
}
