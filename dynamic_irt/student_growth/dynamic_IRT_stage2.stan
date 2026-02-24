data {
  int<lower=1> N;           // total observations in this batch
  int<lower=1> N_persons;   // number of persons in this batch
  int<lower=1> N_items;
  int<lower=1> T_max;       // max time index in this batch
  array[N] int<lower=1,upper=N_persons> person;
  array[N] int<lower=1,upper=N_items>   item;
  array[N] int<lower=1,upper=T_max>     time;
  array[N] int<lower=0,upper=1>         response;

  // Fixed item parameters from Stage 1
  vector[N_items] beta_fixed;
  real<lower=0> sigma_fixed;
}

parameters {
  vector[N_persons] theta_0;
  vector[N_persons] theta_growth;
  matrix[N_persons, T_max] eta;
}

transformed parameters {
  matrix[N_persons, T_max] theta;
  for (p in 1:N_persons)
    for (t in 1:T_max)
      theta[p, t] = theta_0[p] + theta_growth[p] * t + sigma_fixed * eta[p, t];
}

model {
  theta_0       ~ normal(0, 1);
  theta_growth  ~ normal(0, 0.01);
  to_vector(eta) ~ normal(0, 1);

  for (n in 1:N)
    response[n] ~ bernoulli_logit(
      theta[person[n], time[n]] - beta_fixed[item[n]]
    );
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = bernoulli_logit_lpmf(
      response[n] | theta[person[n], time[n]] - beta_fixed[item[n]]
    );
}
