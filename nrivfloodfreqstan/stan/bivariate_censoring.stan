
/**
* Bi-variate 
*
*  Throughout the code:
*   - y variable is obs
*   - z variable is covariate
*
*  Marginal codes:
*   - 1=Gumbel
*   - 2=LogNormal
*   - 3=GEV
*   - 4=LogPearson3
*   - 5=LogPearson3
*
*  Copula codes:
*   - 1=Gumbel
*   - 2=Clayton
*   - 3=Gaussian
*
*  The 6 different cases depending on data availability are coded as follows:
*  11: y and z observed         12: y observed, z censored
*  21: y censored, z observed   22: y censored, z censored
*  31: y missing, z observed    32: y missing, z censored
*
**/


functions {

    #include marginal.stanfunctions
    #include copula.stanfunctions

}


data {
  // Defines marginal distributions
  // 1=Gumbel, 2=LogNormal, 3=GEV, 4=LogPearson3, 5=Normal
  int<lower=1, upper=5> ymarginal; 
  int<lower=1, upper=5> zmarginal; 

  // Defines copula model
  // 1=Gumbel, 2=Clayton, 3=Gaussian
  int<lower=1, upper=3> copula; 

  int N; // total number of values
  vector[N] y; // Data for first variable (ams streamflow)
  vector[N] z; // Data for second variable (ams AWRA)

  // Indexing
  // .. number of values - 6 cases (see comment above)
  array[3, 2] int<lower=0, upper=N> Ncases;

  // cases where covariate is available
  array[Ncases[1,1]] int<lower=1, upper=N> i11;
  array[Ncases[2,1]] int<lower=1, upper=N> i21;
  array[Ncases[3,1]] int<lower=1, upper=N> i31;

  // cases where covariate is censored
  array[Ncases[1,2]] int<lower=1, upper=N> i12;
  array[Ncases[2,2]] int<lower=1, upper=N> i22;
  array[Ncases[3,2]] int<lower=1, upper=N> i32;

  // Prior parameters
  vector[2] yloc_prior;
  vector[2] ylogscale_prior;
  vector[2] yshape_prior;
  
  vector[2] zloc_prior;
  vector[2] zlogscale_prior;
  vector[2] zshape_prior;
  
  real shape_lower;
  real<lower=shape_lower> shape_upper;

  vector[2] rho_prior; 
  real<lower=-0.999, upper=0.998> rho_lower;
  real<lower=rho_lower, upper=0.999> rho_upper;

  // Censoring thresholds 
  real<lower=0> ycensor;
  real<lower=0> zcensor;
}

parameters {
  // Parameter for observed streamflow
  real yloc; 
  real ylogscale;
  real yshape;

  // Parameter for covariate
  real zloc; 
  real zlogscale;
  real zshape;

  // Parameter for copula correlation
  real rho;
}  


transformed parameters {
  real yscale = exp(ylogscale);
  real zscale = exp(zlogscale);

  // Reduced censor
  real ucensor = marginal_cdf(ycensor | ymarginal, yloc, yscale, yshape);
  real vcensor = marginal_cdf(zcensor | zmarginal, zloc, zscale, zshape);
  // .. need a matrix to pass to copula cdf
  matrix[1, 2] uvcensors = [[ucensor, vcensor]];

  // Reduced variable for y and z
  matrix[N, 2] uv;
  
  // .. cases with y observed
  for(i in 1:Ncases[1, 1])
    uv[i11[i], 1] = marginal_cdf(y[i11[i]] | ymarginal, yloc, yscale, yshape);

  for(i in 1:Ncases[1, 2])
    uv[i12[i], 1] = marginal_cdf(y[i12[i]] | ymarginal, yloc, yscale, yshape);

  // .. cases with z observed
  for(i in 1:Ncases[1, 1])
    uv[i11[i], 2] = marginal_cdf(z[i11[i]] | zmarginal, zloc, zscale, zshape);

  for(i in 1:Ncases[2, 1])
    uv[i21[i], 2] = marginal_cdf(z[i21[i]] | zmarginal, zloc, zscale, zshape);

  for(i in 1:Ncases[3, 1])
    uv[i31[i], 2] = marginal_cdf(z[i31[i]] | zmarginal, zloc, zscale, zshape);
}


model {
  // --- Priors --
  yloc ~ normal(yloc_prior[1], yloc_prior[2]);
  ylogscale ~ normal(ylogscale_prior[1], ylogscale_prior[2]);
  yshape ~ normal(yshape_prior[1], yshape_prior[2]) T[shape_lower, shape_upper];

  zloc ~ normal(yloc_prior[1], yloc_prior[2]);
  zlogscale ~ normal(ylogscale_prior[1], ylogscale_prior[2]);
  zshape ~ normal(zshape_prior[1], zshape_prior[2]) T[shape_lower, shape_upper];

  rho ~ normal(rho_prior[1], rho_prior[2]) T[rho_lower, rho_upper];

  // --- Copula likelihood : 6 cases---
  //  11: y and z observed         12: y observed, z censored
  //  21: y censored, z observed   22: y censored, z censored
  //  31: y missing, z observed    32: y missing, z censored

  // Case 11 : both y and z observed
  target += copula_lpdf(uv[i11,:] | copula, rho);
  target += marginal_lpdf(y[i11] | ymarginal, yloc, yscale, yshape);
  target += marginal_lpdf(z[i11] | zmarginal, zloc, zscale, zshape);

  // Case 21 : y censored and z observed
  target += copula_lpdf_ucensored(ucensor, uv[i21, 2], copula, rho);
  target += marginal_lpdf(z[i21] | zmarginal, zloc, zscale, zshape);

  // Case 12 : y observed and z censored
  // we re-use the expression of case 12 and invert u and v variables
  target += copula_lpdf_ucensored(vcensor, uv[i12, 1], copula, rho);
  target += marginal_lpdf(y[i12] | ymarginal, yloc, yscale, yshape);

  // Case 22 : both y and z censored. Copulas cdf times 
  // the number of occurences for 22
  target += Ncases[2,2]*copula_lcdf(uvcensors | copula, rho);

  // Case 31 : y is missing and z is observed. 
  target += marginal_lpdf(z[i31] | zmarginal, zloc, zscale, zshape);

  // Case 32 : y is missing and z is censored 
  // this a Gumbel cdf for z reduced variable times
  // the number of occurences for 32
  target += Ncases[3,2]*marginal_lcdf(zcensor | zmarginal, zloc, zscale, zshape);

}


