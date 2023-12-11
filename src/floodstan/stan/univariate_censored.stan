
/**
* Univariate
*
*  Throughout the code:
*   - Obs variable is y
*
*  Marginal codes:
*   - 1=Gumbel
*   - 2=LogNormal
*   - 3=GEV
*   - 4=LogPearson3
*   - 5=Normal
*   - 6=Generalized Pareto
*
*  The 2 different cases depending on data availability are coded as follows:
*  11: y observed         
*  21: y censored
*  Other cases described in bivariate_censoring.stan are not used.
*
**/


functions {

    #include marginal.stanfunctions

}


data {
  // Defines marginal distributions
  // 1=Gumbel, 2=LogNormal, 3=GEV, 4=LogPearson3, 5=Normal
  // 6=Gen Pareto, 7=Gen Logistic, 8=Gamma
  int<lower=1, upper=8> ymarginal; 

  int N; // total number of values
  vector[N] y; // Data for first variable (ams streamflow)

  // Indexing
  // .. number of values - 9 cases (only 2 used in univariate fitting)
  array[3, 3] int<lower=0, upper=N> Ncases;

  // 2 cases obs/censored
  array[Ncases[1,1]] int<lower=1, upper=N> i11;
  array[Ncases[2,1]] int<lower=1, upper=N> i21;

  // Prior parameters
  vector[2] ylocn_prior;
  
  real<lower=-10> logscale_lower;
  real<lower=logscale_lower, upper=20> logscale_upper;
  vector[2] ylogscale_prior;
  
  vector[2] yshape1_prior;
  
  real shape1_lower;
  real<lower=shape1_lower> shape1_upper;

  // Censoring thresholds 
  real<lower=0> ycensor;
}

parameters {
  // Parameter for observed streamflow
  real ylocn; 
  real ylogscale;
  real yshape1;
}  


transformed parameters {
  real yscale = exp(ylogscale);
}


model {
  // --- Priors --
  ylocn ~ normal(ylocn_prior[1], ylocn_prior[2]);
  ylogscale ~ normal(ylogscale_prior[1], ylogscale_prior[2]) T[logscale_lower, logscale_upper];
  yshape1 ~ normal(yshape1_prior[1], yshape1_prior[2]) T[shape1_lower, shape1_upper];

  // --- Copula likelihood : 6 cases---
  //  11: y observed
  //  21: y censored

  // Case 11 : y observed
  target += marginal_lpdf(y[i11] | ymarginal, ylocn, yscale, yshape1);

  // Case 21 : y censored 
  target += Ncases[2,1]*marginal_lcdf(ycensor | ymarginal, ylocn, yscale, yshape1);
}


