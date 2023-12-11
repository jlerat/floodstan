
/**
* Count modelling
*
*  Throughout the code:
*   - Obs variable is k
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
  int<lower=1, upper=3> kdisc; 

  int<lower=1> N; // total number of values
  int<lower=1> nevent_upper; // total number of values
  array[N] int<lower=0, upper=nevent_upper> k; // Count data 

  // Prior parameters
  real<lower=1> locn_upper;
  vector[2] klocn_prior;
  
  real<lower=0> phi_lower;
  real<lower=phi_lower> phi_upper;
  vector[2] kphi_prior;
}

parameters {
  real<lower=0, upper=locn_upper> klocn; 
  real<lower=phi_lower, upper=phi_upper> kphi;
}  


model {
  // --- Priors --
  klocn ~ normal(klocn_prior[1], klocn_prior[2]) T[0, locn_upper];
  kphi ~ normal(kphi_prior[1], kphi_prior[2]) T[phi_lower, phi_upper];

  for(i in 1:N) {
    target += discrete_lpmf(k[i] | kdisc, klocn, kphi);
  }
}


