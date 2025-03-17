functions {

    #include logpearson3.stanfunctions

}


data {
  int N; // total number of values
  vector[N] y; // Data for first variable (ams streamflow)

  // Indexing
  // .. number of values - 9 cases (only 2 used in univariate fitting)
  array[3, 3] int<lower=0, upper=N> Ncases;

  // 2 cases obs/censored
  array[Ncases[1,1]] int<lower=1, upper=N> i11;
  array[Ncases[2,1]] int<lower=1, upper=N> i21;

  // Prior parameters
  real<lower=-1e10> locn_lower;
  real<lower=locn_lower, upper=1e10> locn_upper;
  vector[2] ylocn_prior;
  
  real<lower=-20> logscale_lower;
  real<lower=logscale_lower, upper=20> logscale_upper;
  vector[2] ylogscale_prior;
  
  vector[2] yshape1_prior;
  
  real shape1_lower;
  real<lower=shape1_lower> shape1_upper;

  // Censoring thresholds 
  real ycensor;
}

parameters {
  real ylocn; 
  real ylogscale;
  real yshape1;
}  


transformed parameters {
  real yscale = exp(ylogscale);
}


model {
  // --- Priors --
  ylocn ~ normal(ylocn_prior[1], ylocn_prior[2]) T[locn_lower, locn_upper];
  ylogscale ~ normal(ylogscale_prior[1], ylogscale_prior[2]) T[logscale_lower, logscale_upper];
  yshape1 ~ normal(yshape1_prior[1], yshape1_prior[2]) T[shape1_lower, shape1_upper];

  // --- Copula likelihood : 6 cases---
  //  11: y observed
  //  21: y censored

  // Case 11 : y observed
  target += logpearson3_lpdf(y[i11] | ylocn, yscale, yshape1);

  // Case 21 : y censored 
  if(Ncases[2, 1] > 0)
     target += Ncases[2,1] * logpearson3_lcdf(ycensor | ylocn, yscale, yshape1);
}
