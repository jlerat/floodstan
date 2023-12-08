
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

    #include count.stanfunctions

}


data {
  // Defines count distributions
  // 1=Poisson, 2=LogNormal
  int<lower=1, upper=2> ycount; 

  int<lower=1> N; // total number of values
  array[N] int<lower=0> yn; // Count data 

  // Prior parameters
  vector[2] ylocn_prior;
  
  real<lower=0> phi_lower;
  real<lower=phi_lower> phi_upper;
  vector[2] yphi_prior;
}

parameters {
  real ylocn; 
  real yphi;
}  


model {
  // --- Priors --
  ylocn ~ normal(ylocn_prior[1], ylocn_prior[2]) T[0,];
  yphi ~ normal(yphi_prior[1], yphi_prior[2]) T[phi_lower, phi_upper];

  for(i in 1:N) {
    target += count_lpmf(yn[i] | ycount, ylocn, yphi);
  }
}


