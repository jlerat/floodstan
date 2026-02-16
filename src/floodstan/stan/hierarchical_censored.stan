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
    array[M] int<lower=5,upper=N> Nobs;
    array[M, N] int<lower=0,upper=N> idx_obs;
    
    array[M] int<lower=0,upper=N> Ncens;

    // Streamflow AMS data
    array[M] vector[N] y; 
    
    // Censoring thresholds
    vector[M] ycensors; 

    // Priors
    vector[3] rho_lower;
    vector[3] rho_upper;
    array[3] vector[2] rho_prior;

    vector[3] alpha_lower;
    vector[3] alpha_upper;
    array[3] vector[2] alpha_prior;

    vector[3] sigma_lower;
    vector[3] sigma_upper;
    array[3] vector[2] sigma_prior;
   
    vector[3] beta0_lower;
    vector[3] beta0_upper;
    array[3] vector[2] beta0_prior;

    vector[3] beta1_lower;
    vector[3] beta1_upper;
    array[3] vector[2] beta1_prior;
}  

transformed data {
    vector[M] log_areas = log(areas);
    vector[M] ones = rep_vector(1., M);

    // Compute diff to ensure that lower < upper
    vector[M] rho_diff;
    vector[M] alpha_diff;
    vector[M] sigma_diff;
    vector[M] beta0_diff;
    vector[M] beta1_diff;

    real<lower=1e-3> diff = 2e-3;

    for(iv in 1:3) {
        diff = rho_upper[iv] - rho_lower[iv];
        rho_diff[iv] = diff;
        
        diff = alpha_upper[iv] - alpha_lower[iv];
        alpha_diff[iv] = diff;
        
        diff = sigma_upper[iv] - sigma_lower[iv];
        sigma_diff[iv] = diff;

        diff = beta0_upper[iv] - beta0_lower[iv];
        beta0_diff[iv] = diff;

        diff = beta1_upper[iv] - beta1_lower[iv];
        beta1_diff[iv] = diff;
    }
}

parameters {
   // Marginal parameters
   vector[M] yasinhlocn;        
   vector[M] ylogscale;        
   vector[M] yshape1;        

   // Normalised GP parameter
   vector<lower=0, upper=1>[3] u_rho;
   vector<lower=0, upper=1>[3] u_alpha;
   vector<lower=0, upper=1>[3] u_sigma;
   
   // Normalised regression parameters
   vector<lower=0, upper=1>[3] u_beta0;
   vector<lower=0, upper=1>[2] u_beta1;
}

transformed parameters {
   vector[M] ylocn = sinh(yasinhlocn);
   vector[M] yscale = exp(ylogscale);

    
   vector[3] rho;
   vector[3] alpha;
   vector[3] sigma;
   vector[3] beta0;
   vector[2] beta1;

   for(iv in 1:3) {
        rho[iv] = rho_lower[iv] + rho_diff[iv] * u_rho[iv];
        alpha[iv] = alpha_lower[iv] + alpha_diff[iv] * u_alpha[iv];
        sigma[iv] = sigma_lower[iv] + sigma_diff[iv] * u_sigma[iv];
        
        beta0[iv] = beta0_lower[iv] + beta0_diff[iv] * u_beta0[iv];
        if(iv < 3)
            beta1[iv] = beta1_lower[iv] + beta1_diff[iv] * u_beta1[iv];
   }
}

model {
    // See https://mc-stan.org/docs/stan-users-guide/gaussian-processes.html#fit-gp.section

    // Covariance - Exponential kernel
    matrix [M, M] Klocn = gp_exponential_cov(coords, alpha[1], rho[1]);
    matrix [M, M] Klogs = gp_exponential_cov(coords, alpha[2], rho[2]);
    matrix [M, M] Kshp = gp_exponential_cov(coords, alpha[3], rho[3]);

    // Adding the nugget
    for(ista in 1:M) {
        Klocn[ista, ista] = Klocn[ista, ista] + square(sigma[1]);
        Klogs[ista, ista] = Klogs[ista, ista] + square(sigma[2]);
        Kshp[ista, ista] = Kshp[ista, ista] + square(sigma[3]);
    }    

    matrix[M, M] Llocn = cholesky_decompose(Klocn);
    matrix[M, M] Llogs = cholesky_decompose(Klogs);
    matrix[M, M] Lshp = cholesky_decompose(Kshp);

    // Priors
    for(iv in 1:3){
        rho[iv] ~ normal(rho_prior[iv][1], rho_prior[iv][2]);
        alpha[iv] ~ normal(alpha_prior[iv][1], alpha_prior[iv][2]);
        sigma[iv] ~ normal(sigma_prior[iv][1], sigma_prior[iv][2]); 

        beta0[iv] ~ normal(beta0_prior[iv][1], beta0_prior[iv][2]);

        if(iv < 3) 
            beta1[iv] ~ normal(beta1_prior[iv][1], beta1_prior[iv][2]);
    }

    // Likelihood
    yasinhlocn ~ multi_normal_cholesky(beta0[1] + beta1[1] * log_areas, Llocn);
    ylogscale ~ multi_normal_cholesky(beta0[2] + beta1[2] * log_areas, Llogs);
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

