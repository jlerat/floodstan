
/**
* Test functions

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

  // Censoring thresholds 
  real<lower=0> ycensor;
  real<lower=0> zcensor;
  
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


generated quantities {
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

  // --- Copula likelihood : 6 cases---
  //  11: y and z observed         12: y observed, z censored
  //  21: y censored, z observed   22: y censored, z censored
  //  31: y missing, z observed    32: y missing, z censored

  // Case 11 : both y and z observed
  real l11_a = copula_lpdf(uv[i11,:] | copula, rho);
  real l11_b = marginal_lpdf(y[i11] | ymarginal, yloc, yscale, yshape);
  real l11_c = marginal_lpdf(z[i11] | zmarginal, zloc, zscale, zshape);

  // Case 21 : y censored and z observed
  real l21_a = copula_lpdf_ucensored(ucensor, uv[i21, 2], copula, rho);
  real l21_b = marginal_lpdf(z[i21] | zmarginal, zloc, zscale, zshape);

  // Case 12 : y observed and z censored
  // we re-use the expression of case 12 and invert u and v variables
  real l12_a = copula_lpdf_ucensored(vcensor, uv[i12, 1], copula, rho);
  real l12_b = marginal_lpdf(y[i12] | ymarginal, yloc, yscale, yshape);

  // Case 22 : both y and z censored. Copulas cdf times 
  // the number of occurences for 22
  real l22 = Ncases[2,2]*copula_lcdf(uvcensors | copula, rho);

  // Case 31 : y is missing and z is observed. 
  real l31 = marginal_lpdf(z[i31] | zmarginal, zloc, zscale, zshape);

  // Case 32 : y is missing and z is censored 
  // this a Gumbel cdf for z reduced variable times
  // the number of occurences for 32
  real l32 = Ncases[3,2]*marginal_lcdf(zcensor | zmarginal, zloc, zscale, zshape);

}


