
/**
* Test functions

**/


functions {

    #include marginal.stanfunctions

}


data {
  // Defines marginal distributions
  // 1=Gumbel, 2=LogNormal, 3=GEV, 4=LogPearson3, 5=Normal
  int<lower=1, upper=5> ymarginal; 

  int<lower=1> N; // total number of values
  vector[N] y; // Data for first variable (ams streamflow)

  // Parameter for observed streamflow
  real ylocn; 
  real ylogscale;
  real yshape1;

}  


generated quantities {
  real ylocn_check = ylocn;
  real ylogscale_check = ylogscale;
  real yshape1_check = yshape1;

  real yscale = exp(ylogscale);

  // un censored case
  vector[N] luncens;
  vector[N] lcens;
  vector[1] tmp;

  for(i in 1:N){
    tmp[1] = y[i];
    luncens[i] = marginal_lpdf(tmp | ymarginal, ylocn, yscale, yshape1);
    lcens[i] = marginal_lcdf(y[i] | ymarginal, ylocn, yscale, yshape1);
  }


}


