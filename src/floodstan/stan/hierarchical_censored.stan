functions {

    #include marginal.stanfunctions

    matrix gp_cov_cholesky(array[] vector coords, real sigma2, real alpha2, real rho){
        int M = size(coords);
        matrix [M, M] cov;
        cov =  gp_exponential_cov(coords, sqrt(alpha2), rho);
        return cholesky_decompose(add_diag(cov, sigma2));
    }    
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
    real<lower=0> rho_lower;
    real<lower=rho_lower> rho_upper;
    vector[2] rho_prior;

    vector<lower=0>[3] tau2_lower;
    vector[3] tau2_upper;
    array[3] vector[2] tau2_prior;
   
    vector<lower=0, upper=1>[3] u_alpha2;

    vector[3] beta0_lower;
    vector[3] beta0_upper;
    array[3] vector[2] beta0_prior;

    vector<lower=0>[2] beta1_lower;
    vector[2] beta1_upper;
    array[2] vector[2] beta1_prior;

    int<lower=0, upper=1> shape_has_hierarchical;
}  

transformed data {
    vector[M] logareas = log(areas);
    vector[M] ones = rep_vector(1., M);

    // Check that upper > lower
    real<lower=1e-5> check_tau2 = min(tau2_upper - tau2_lower);
    real<lower=1e-5> check_beta0 = min(beta0_upper - beta0_lower);
    real<lower=1e-5> check_beta1 = min(beta1_upper - beta1_lower);
}

parameters {
   // Marginal parameters
   vector[M] yloglocn;        
   vector[M] ylogscale;        
   vector[M] yshape1;        

   // Normalised GP parameter
   vector<lower=rho_lower, upper=rho_upper>[3] rho;
   vector<lower=0, upper=1>[3] u_tau2;
   
   // Normalised regression parameters
   vector<lower=0, upper=1>[3] u_beta0;
   vector<lower=0, upper=1>[2] u_beta1;
}

transformed parameters {
   vector[M] ylocn = exp(yloglocn);
   vector[M] yscale = exp(ylogscale);

   vector[3] tau2 = tau2_lower + (tau2_upper - tau2_lower) .* u_tau2;
   vector[3] beta0 = beta0_lower + (beta0_upper - beta0_lower) .* u_beta0;
   vector[2] beta1 = beta1_lower + (beta1_upper - beta1_lower) .* u_beta1;
   
   vector[3] sigma2 = tau2 .* (1 - u_alpha2);
   vector[3] alpha2 = tau2 .* u_alpha2;
}


model {
    // See https://mc-stan.org/docs/stan-users-guide/gaussian-processes.html#fit-gp.section

    // Covariance - Exponential kernel
    matrix [M, M] Llocn = gp_cov_cholesky(coords, sigma2[1], alpha2[1], rho[1]);
    matrix [M, M] Llogs = gp_cov_cholesky(coords, sigma2[2], alpha2[2], rho[2]);
    matrix [M, M] Lshp = gp_cov_cholesky(coords, sigma2[3], alpha2[3], rho[3]);

    // Priors
    rho ~ normal(rho_prior[1], rho_prior[2]);

    for(iv in 1:3){
        tau2[iv] ~ normal(tau2_prior[iv][1], tau2_prior[iv][2]);

        beta0[iv] ~ normal(beta0_prior[iv][1], beta0_prior[iv][2]);
        if(iv < 3) 
            beta1[iv] ~ normal(beta1_prior[iv][1], beta1_prior[iv][2]);
    }

    // Likelihood
    yloglocn ~ multi_normal_cholesky(beta0[1] + beta1[1] * logareas, Llocn);
    ylogscale ~ multi_normal_cholesky(beta0[2] + beta1[2] * logareas, Llogs);

    if(shape_has_hierarchical == 1)
        yshape1 ~ multi_normal_cholesky(beta0[3] * ones, Lshp);
    else
        yshape1 ~ normal(beta0_prior[3][1], beta0_prior[3][2]);
        
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
    // Square root parameters
    vector[3] sigma = sqrt(sigma2);
    vector[3] alpha = sqrt(alpha2);

    // Sample from hierarchical prior
    vector[M] ylocn_hprior;
    vector[M] ylogscale_hprior;
    vector[M] yshape1_hprior;
    {
        // Covariance matrix and arcsinh transformed location paramters 
        // are declared as 'local' variables to avoid storing them in 
        // the output files
        matrix [M, M] Llocn = gp_cov_cholesky(coords, sigma2[1], alpha2[1], rho[1]);
        matrix [M, M] Llogs = gp_cov_cholesky(coords, sigma2[2], alpha2[2], rho[2]);
        matrix [M, M] Lshp = gp_cov_cholesky(coords, sigma2[3], alpha2[3], rho[3]);

        vector[M] yloglocn_hprior = multi_normal_cholesky_rng(beta0[1] + beta1[1] * logareas, Llocn);
        ylocn_hprior = exp(yloglocn_hprior);
        ylogscale_hprior = multi_normal_cholesky_rng(beta0[2] + beta1[2] * logareas, Llogs);

        if(shape_has_hierarchical == 1)
            yshape1_hprior = multi_normal_cholesky_rng(beta0[3] * ones, Lshp);
        else
            yshape1_hprior = to_vector(normal_rng(beta0_prior[3][1] * ones, beta0_prior[3][2]));
    }
}

