
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
*
*  The 2 different cases depending on data availability are coded as follows:
*  11: y observed         
*  21: y censored
*
**/


functions {

    #include marginal.stanfunctions

}


data {
  // Defines marginal distributions
  // 1=Gumbel, 2=LogNormal, 3=GEV, 4=LogPearson3, 5=Normal
  int<lower=1, upper=5> ymarginal; 

  int N; // total number of values
  vector[N] y; // Data for first variable (ams streamflow)

  // Indexing
  // .. number of values - 6 cases (only 2 used in univariate fitting)
  array[3, 2] int<lower=0, upper=N> Ncases;

  // 2 cases obs/censored
  array[Ncases[1,1]] int<lower=1, upper=N> i11;
  array[Ncases[2,1]] int<lower=1, upper=N> i21;

  // Prior parameters
  vector[2] yloc_prior;
  vector[2] ylogscale_prior;
  vector[2] yshape_prior;
  
  real shape_lower;
  real<lower=shape_lower> shape_upper;

  // Censoring thresholds 
  real<lower=0> ycensor;
}

parameters {
  // Parameter for observed streamflow
  real yloc; 
  real ylogscale;
  real yshape;
}  


transformed parameters {
  real yscale = exp(ylogscale);
}


model {
  // --- Priors --
  yloc ~ normal(yloc_prior[1], yloc_prior[2]);
  ylogscale ~ normal(ylogscale_prior[1], ylogscale_prior[2]);
  yshape ~ normal(yshape_prior[1], yshape_prior[2]) T[shape_lower, shape_upper];

  // --- Copula likelihood : 6 cases---
  //  11: y observed
  //  21: y censored

  // Case 11 : y observed
  target += marginal_lpdf(y[i11] | ymarginal, yloc, yscale, yshape);

  // Case 21 : y censored 
  target += Ncases[2,1]*marginal_lcdf(ycensor | ymarginal, yloc, yscale, yshape);
}


