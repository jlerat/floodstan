
/**
* Test functions

**/


functions {

    #include discrete.stanfunctions

}


data {
  // Defines count distributions
  // 1=Poisson, 2=NegativeBinomial, 3=Bernouilli
  int<lower=1, upper=3> edisc; 

  int<lower=1> N; // total number of values
  int<lower=1> nevent_upper; // total number of values
  array[N] int<lower=0, upper=nevent_upper> e; // Count data 

  // Parameter for observed streamflow
  real elocn; 
  real ephi;
}  


generated quantities {
  real elocn_check = elocn;
  real ephi_check = ephi;

  // un censored case
  vector[N] lpmf;
  for(i in 1:N) {
    lpmf[i] = discrete_lpmf(e[i] | edisc, elocn, ephi);
  }  
}


