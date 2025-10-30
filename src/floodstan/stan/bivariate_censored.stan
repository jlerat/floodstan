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
*   - 5=Normal
*   - 6=Generalized Pareto
*   - 7=Generalized Logistic
*   - 8=Gamma
*
*  Copula codes:
*   - 1=Gumbel
*   - 2=Clayton
*   - 3=Gaussian
*
*  The 9 different cases depending on data availability are coded as follows:
*  11: y and z observed         12: y observed, z censored   13: y observed, z missing
*  21: y censored, z observed   22: y censored, z censored   23: y censored, z missing
*  31: y missing, z observed    32: y missing, z censored    33: y missing, z missing
*
**/

functions {

    #include marginal.stanfunctions
    #include copula.stanfunctions

}

data {
  // Defines marginal distributions
  int<lower=1, upper=8> ymarginal; 
  int<lower=1, upper=8> zmarginal; 

  // Defines copula model
  // 1=Gumbel, 2=Clayton, 3=Gaussian
  int<lower=1, upper=3> copula; 

  int N; // total number of values
  vector[N] y; // Data for first variable (ams streamflow)
  vector[N] z; // Data for second variable (ams AWRA)

  // Indexing
  // .. number of values - 9 cases (see comment above)
  array[3, 3] int<lower=0, upper=N> Ncases;

  // cases where covariate is available
  array[Ncases[1,1]] int<lower=1, upper=N> i11;
  array[Ncases[2,1]] int<lower=1, upper=N> i21;
  array[Ncases[3,1]] int<lower=1, upper=N> i31;

  // cases where covariate is censored
  array[Ncases[1,2]] int<lower=1, upper=N> i12;
  array[Ncases[2,2]] int<lower=1, upper=N> i22;
  array[Ncases[3,2]] int<lower=1, upper=N> i32;

  // cases where covariate is missing
  array[Ncases[1,3]] int<lower=1, upper=N> i13;
  array[Ncases[2,3]] int<lower=1, upper=N> i23;

  // Prior parameters
  real<lower=-1e10> locn_lower;
  real<lower=locn_lower, upper=1e10> locn_upper;
  vector[2] ylocn_prior;
  
  real<lower=-20> logscale_lower;
  real<lower=logscale_lower, upper=20> logscale_upper;
  vector[2] ylogscale_prior;
  
  vector[2] yshape1_prior;
  
  vector[2] zlocn_prior;
  vector[2] zlogscale_prior;
  vector[2] zshape1_prior;
  
  real shape1_lower;
  real<lower=shape1_lower> shape1_upper;

  vector[2] rho_prior; 
  real<lower=0.001, upper=0.998> rho_lower;
  real<lower=rho_lower, upper=0.999> rho_upper;

  // Censoring thresholds 
  real ycensor;
  real zcensor;
}

transformed data {
  // Imposes that at least 2 data points are observed
  // for variable y
  int<lower=2> Ny = Ncases[1, 1] + Ncases[1, 2] + Ncases[1, 3];  
  int<lower=2> Nz = Ncases[1, 1] + Ncases[2, 1] + Ncases[3, 1];  

  // Sanity check on number of samples
  int<lower=0, upper=0> Ncheck = sum(Ncases) - N;
}

parameters {
  // Parameter for observed streamflow
  // .. no bounds for loc, see univariate_censored_sampling
  real ylocn;
  real<lower=logscale_lower, upper=logscale_upper> ylogscale;
  real<lower=shape1_lower, upper=shape1_upper> yshape1;

  // Parameter for covariate
  real zlocn;
  real<lower=logscale_lower, upper=logscale_upper> zlogscale;
  real<lower=shape1_lower, upper=shape1_upper> zshape1;

  // Parameter for copula correlation
  real<lower=rho_lower, upper=rho_upper> rho;
}  


transformed parameters {
  real yscale = exp(ylogscale);
  real zscale = exp(zlogscale);

  // Reduced censor
  real ucensor = marginal_cdf(ycensor | ymarginal, ylocn, yscale, yshape1);
  real vcensor = marginal_cdf(zcensor | zmarginal, zlocn, zscale, zshape1);
  // .. need a matrix to pass to copula cdf
  matrix[1, 2] uvcensors = [[ucensor, vcensor]];

  // Reduced variable for y and z
  matrix[N, 2] uv;
  
  // .. cases with y and z observed
  for(i in 1:Ncases[1, 1]) {
    uv[i11[i], 1] = marginal_cdf(y[i11[i]] | ymarginal, ylocn, yscale, yshape1);
    uv[i11[i], 2] = marginal_cdf(z[i11[i]] | zmarginal, zlocn, zscale, zshape1);
  }

  // .. cases with y observed only
  for(i in 1:Ncases[1, 2])
    uv[i12[i], 1] = marginal_cdf(y[i12[i]] | ymarginal, ylocn, yscale, yshape1);
  
  for(i in 1:Ncases[1, 3])
    uv[i13[i], 1] = marginal_cdf(y[i13[i]] | ymarginal, ylocn, yscale, yshape1);

  // .. cases with z observed only
  for(i in 1:Ncases[2, 1])
    uv[i21[i], 2] = marginal_cdf(z[i21[i]] | zmarginal, zlocn, zscale, zshape1);

  for(i in 1:Ncases[3, 1])
    uv[i31[i], 2] = marginal_cdf(z[i31[i]] | zmarginal, zlocn, zscale, zshape1);
}


model {
  // --- Priors --
  ylocn ~ normal(ylocn_prior[1], ylocn_prior[2]) T[locn_lower, locn_upper];
  ylogscale ~ normal(ylogscale_prior[1], ylogscale_prior[2]) T[logscale_lower, logscale_upper];
  yshape1 ~ normal(yshape1_prior[1], yshape1_prior[2]) T[shape1_lower, shape1_upper];

  zlocn ~ normal(ylocn_prior[1], ylocn_prior[2]) T[locn_lower, locn_upper];
  zlogscale ~ normal(ylogscale_prior[1], ylogscale_prior[2]) T[logscale_lower, logscale_upper];
  zshape1 ~ normal(zshape1_prior[1], zshape1_prior[2]) T[shape1_lower, shape1_upper];

  rho ~ normal(rho_prior[1], rho_prior[2]) T[rho_lower, rho_upper];

  // --- Copula likelihood : 9 cases---
  //  11: y and z observed        12: y observed, z censored  13: y obs, z miss
  //  21: y censored, z observed  22: y censored, z censored  23: y cens, z miss
  //  31: y missing, z observed   32: y missing, z censored   33: y miss, z miss
  // 
  // All cases are covered below except 33 where there is not data for both
  // y and z. This case does not contribute to the likelihood.

  // Case 11 : both y and z observed
  if(Ncases[1, 1] > 0) {
    target += copula_lpdf(uv[i11,:] | copula, rho);
    target += marginal_lpdf(y[i11] | ymarginal, ylocn, yscale, yshape1);
    target += marginal_lpdf(z[i11] | zmarginal, zlocn, zscale, zshape1);
  }

  // Case 21 : y censored and z observed
  if(Ncases[2, 1] > 0) {
    target += copula_lpdf_conditional(uv[i21, 2], ucensor, copula, rho);
    target += marginal_lpdf(z[i21] | zmarginal, zlocn, zscale, zshape1);
  }

  // Case 31 : y is missing and z is observed. 
  // this is marginal pdf for z
  if(Ncases[3, 1] > 0) 
    target += marginal_lpdf(z[i31] | zmarginal, zlocn, zscale, zshape1);

  // Case 12 : y observed and z censored
  // we re-use the expression of case 12 and invert u and v variables
  if(Ncases[1, 2] > 0) {
    target += copula_lpdf_conditional(uv[i12, 1], vcensor, copula, rho);
    target += marginal_lpdf(y[i12] | ymarginal, ylocn, yscale, yshape1);
  }

  // Case 22 : both y and z censored. Copulas cdf times 
  // the number of occurences for 22
  if(Ncases[2, 2] > 0)
     target += Ncases[2, 2]*copula_lcdf(uvcensors | copula, rho);

  // Case 32 : y is missing and z is censored 
  // this a marginal cdf for z reduced variable times
  // the number of occurences for 32
  if(Ncases[3, 2] > 0)
     target += Ncases[3, 2] * marginal_lcdf(zcensor | zmarginal, zlocn, zscale, zshape1);
  
  // Case 13 : y is obs, z is missing
  // this is marginal pdf for y (analogous to case 31)
  if(Ncases[1, 3] > 0)
     target += marginal_lpdf(y[i13] | ymarginal, ylocn, yscale, yshape1);

  // Case 23 : y is cens, z is missing
  // this is marginal cdf for y (analogous to case 32)
  if(Ncases[2, 3] > 0)
     target += Ncases[2, 3] * marginal_lcdf(ycensor | ymarginal, ylocn, yscale, yshape1);
}


