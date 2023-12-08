
/**
* Test functions

**/


functions {

    #include count.stanfunctions

}


data {
  // Defines count distributions
  // 1=Poisson, 2=NegativeBinomial
  int<lower=1, upper=2> ycount; 

  int<lower=1> N; // total number of values
  array[N] int<lower=0> yn; // Data

  // Parameter for observed streamflow
  real ylocn; 
  real yphi;
}  


generated quantities {
  real ylocn_check = ylocn;
  real yphi_check = yphi;

  // un censored case
  vector[N] lpmf;
  for(i in 1:N) {
    lpmf[i] = count_lpmf(yn[i] | ycount, ylocn, yphi);
  }  
}


