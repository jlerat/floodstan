
/**
* Test functions

**/


functions {

    #include discrete.stanfunctions

}


data {
  // Defines count distributions
  // 1=Poisson, 2=NegativeBinomial, 3=Bernouilli
  int<lower=1, upper=3> kdisc; 

  int<lower=1> N; // total number of values
  int<lower=1> nevent_upper; // total number of values
  array[N] int<lower=0, upper=nevent_upper> k; // Count data 

  // Parameter for observed streamflow
  real klocn; 
  real kphi;
}  


generated quantities {
  real klocn_check = klocn;
  real kphi_check = kphi;

  // un censored case
  vector[N] lpmf;
  for(i in 1:N) {
    lpmf[i] = discrete_lpmf(k[i] | kdisc, klocn, kphi);
  }  
}


