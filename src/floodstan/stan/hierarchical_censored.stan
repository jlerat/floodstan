functions {

    #include marginal.stanfunctions

}

data {
    int<lower=1> N;        // Duration of time series
    int<lower=1> M;        // Number of stations

    vector<lower=1e-2>[M] areas;        // catchment areas
    array[M] vector[2] coords;  // stations coordinates

    // Marginal distributions
    // 1=Gumbel, 2=LogNormal, 3=GEV, 4=LogPearson3, 5=Normal
    // 6=Gen Pareto, 7=Gen Logistic, 8=Gamma
    int<lower=1, upper=8> ymarginal; 

    // Missing and censored data indexes
    array[M] int<lower=2,upper=N> Nobs;
    array[M, N] int<lower=0,upper=N> idx_obs;
    
    array[M] int<lower=0,upper=N> Ncens;

    // Streamflow AMS data
    array[M] vector[N] y; 
    
    // Censoring thresholds
    vector[M] ycensors; 

    // Priors
    real yshape1_lower;
    real<lower=yshape1_lower> yshape1_upper;

    real rho_lower;
    real<lower=rho_lower> rho_upper;
    vector[2] rho_prior;

    real alpha_lower;
    real<lower=alpha_lower> alpha_upper;
    array[3] vector[2] alpha_prior;

    vector[3] beta0_lower;
    vector[3] beta0_upper;
    array[3] vector[2] beta0_prior;

    vector[2] beta1_lower;
    vector[2] beta1_upper;
    array[3] vector[2] beta1_prior;
}  

transformed data {
    vector[M] logareas = log(areas);
    vector[M] ones = rep_vector(1., M);
}

parameters {
   // Marginal parameters
   vector[M] yasinhlocn;        
   vector[M] ylogscale;        
   vector<lower=yshape1_lower, upper=yshape1_upper>[M] yshape1;        

   // Normalised GP parameter
   real<lower=rho_lower, upper=rho_upper> rho;
   vector<lower=alpha_lower, upper=alpha_upper>[3] alpha;
   
   // Normalised regression parameters
   vector<lower=0, upper=1>[3] u_beta0;
   vector<lower=0, upper=1>[2] u_beta1;
}

transformed parameters {
   vector[M] ylocn = sinh(yasinhlocn);
   vector[M] yscale = exp(ylogscale);
   
   vector[3] beta0 = beta0_lower + (beta0_upper - beta0_lower) .* u_beta0;
   vector[2] beta1 = beta1_lower + (beta1_upper - beta1_lower) .* u_beta1;
}

model {
    // See https://mc-stan.org/docs/stan-users-guide/gaussian-processes.html#fit-gp.section

    // Covariance - Exponential kernel
    matrix [M, M] L = cholesky_decompose(gp_exponential_cov(coords, 1., rho));
    matrix [M, M] Llocn = alpha[1] * L;
    matrix [M, M] Llogs = alpha[2] * L;
    matrix [M, M] Lshp = alpha[3] * L;

    // Priors
    rho ~ normal(rho_prior[1], rho_prior[2]);

    for(iv in 1:3){
        alpha[iv] ~ normal(alpha_prior[iv][1], alpha_prior[iv][2]);
        beta0[iv] ~ normal(beta0_prior[iv][1], beta0_prior[iv][2]);
        if(iv < 3) 
            beta1[iv] ~ normal(beta1_prior[iv][1], beta1_prior[iv][2]);
    }

    // Likelihood
    yasinhlocn ~ multi_normal_cholesky(beta0[1] + beta1[1] * logareas, Llocn);
    ylogscale ~ multi_normal_cholesky(beta0[2] + beta1[2] * logareas, Llogs);
    yshape1 ~ multi_normal_cholesky(beta0[3] * ones, Lshp);

    real loc;
    real scale;
    real shape;
    int nobs;
    int ncens;

    for(ista in 1:M) {
          loc = ylocn[ista];
          scale = yscale[ista];
          shape = yshape1[ista];

          // y observed
          nobs = Nobs[ista];
          array[nobs] int iobs = idx_obs[ista, 1:nobs];
          target += marginal_lpdf(y[ista][iobs] | ymarginal, loc, scale, shape);

          // y censored 
          ncens = Ncens[ista];
          if(ncens > 0)
             target += ncens * marginal_lcdf(ycensors[ista] | ymarginal, loc, scale, shape);
    }
}

generated quantities {
    // Sample from hierarchical prior
    vector[M] ylocn_hprior;
    vector[M] ylogscale_hprior;
    vector[M] yshape1_hprior;
    {
        // Covariance matrix and arcsinh transformed location paramters 
        // are declared as 'local' variables to avoid storing them in 
        // the output files
        matrix [M, M] L = cholesky_decompose(gp_exponential_cov(coords, 1., rho[1]));
        matrix [M, M] Llocn = alpha[1] * L;
        matrix [M, M] Llogs = alpha[2] * L;
        matrix [M, M] Lshp =  alpha[3] * L;

        vector[M] yasinhlocn_hprior = multi_normal_cholesky_rng(beta0[1] + beta1[1] * logareas, Llocn);
        ylocn_hprior = sinh(yasinhlocn_hprior);

        ylogscale_hprior = multi_normal_cholesky_rng(beta0[2] + beta1[2] * logareas, Llogs);
        yshape1_hprior = multi_normal_cholesky_rng(beta0[3] * ones, Lshp);
    }
}

