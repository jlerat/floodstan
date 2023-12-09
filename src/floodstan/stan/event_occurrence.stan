
/**
* Count modelling
*
*  Throughout the code:
*   - Obs variable is y
*
*  Count model codes:
*   - 1=Poisson
*   - 2=Negative Binomial
*
**/


functions {

    #include discrete.stanfunctions

}


data {
  // Defines count distributions
  // 1=Poisson, 2=NegBinomial, 3=Bernouilli
  int<lower=1, upper=3> edisc; 

  int<lower=1> N; // total number of values
  int<lower=1> nevent_upper; // total number of values
  array[N] int<lower=0, upper=nevent_upper> e; // Count data 

  // Prior parameters
  vector[2] elocn_prior;
  
  real<lower=0> phi_lower;
  real<lower=phi_lower> phi_upper;
  vector[2] ephi_prior;
}

parameters {
  real elocn; 
  real ephi;
}  


model {
  // --- Priors --
  elocn ~ normal(elocn_prior[1], elocn_prior[2]) T[0,];
  ephi ~ normal(ephi_prior[1], ephi_prior[2]) T[phi_lower, phi_upper];

  for(i in 1:N) {
    target += discrete_lpmf(e[i] | edisc, elocn, ephi);
  }
}


