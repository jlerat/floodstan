
/**
* Test functions

**/


functions {

    #include marginal.stanfunctions

}


data {
  // Defines marginal distributions
  // 1=Gumbel, 2=LogNormal, 3=GEV, 4=LogPearson3, 5=Normal, 6=Generalized Pareto
  // 7=Generalized Logistic, 8=Gamma
  int<lower=1, upper=8> ymarginal; 

  int<lower=1> N; // total number of values
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
  vector[N] cens;
  vector[N] lcens;
  vector[1] tmp;

  for(i in 1:N){
    tmp[1] = y[i];
    luncens[i] = marginal_lpdf(tmp | ymarginal, ylocn, yscale, yshape1);
    cens[i] = marginal_cdf(y[i] | ymarginal, ylocn, yscale, yshape1);
    lcens[i] = marginal_lcdf(y[i] | ymarginal, ylocn, yscale, yshape1);
  }

  // Log-likelihood for univariate censored model
  // Case 11 : y observed
  real loglikelihood = marginal_lpdf(y[i11] | ymarginal, ylocn, yscale, yshape1);

  // Case 21 : y censored 
  if(Ncases[2, 1] > 0)
     loglikelihood += Ncases[2, 1] * marginal_lcdf(ycensor | ymarginal, ylocn, yscale, yshape1);
 
  // Log-priors
  real p0 = normal_cdf(locn_lower | ylocn_prior[1], ylocn_prior[2]);
  real p1 = normal_cdf(locn_upper | ylocn_prior[1], ylocn_prior[2]);
  real lp = normal_lpdf(ylocn | ylocn_prior[1], ylocn_prior[2]);
  real logprior_locn = lp - log(p1 - p0);

  p0 = normal_cdf(logscale_lower | ylogscale_prior[1], ylogscale_prior[2]);
  p1 = normal_cdf(logscale_upper | ylogscale_prior[1], ylogscale_prior[2]);
  lp = normal_lpdf(ylogscale | ylogscale_prior[1], ylogscale_prior[2]);
  real logprior_logscale = lp - log(p1 - p0);

  p0 = normal_cdf(shape1_lower | yshape1_prior[1], yshape1_prior[2]);
  p1 = normal_cdf(shape1_upper | yshape1_prior[1], yshape1_prior[2]);
  lp = normal_lpdf(yshape1 | yshape1_prior[1], yshape1_prior[2]);
  real logprior_shape1 = lp - log(p1 - p0);

  // Log-posterior
  real logposterior = loglikelihood + logprior_locn + logprior_logscale;
  if(ymarginal == 3 || ymarginal == 4 || ymarginal == 6 || ymarginal == 7)
     logposterior += logprior_shape1;
}


