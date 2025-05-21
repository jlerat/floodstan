
data {
    int<lower=1> N;        // Number of data points
    int<lower=0> P;        // Number of predictors
    int<lower=0, upper=N-1> Nvalid;   // Number of valid observations (at least 1 unobserved)

    matrix[N, P] x;        // predictors
    array[N] vector[2] w;  // spatial coordinates
    vector[N] y;           // observed predictands

    array[Nvalid] int<lower=1, upper=N> ivalid;
   
    // Mean parameters
    vector[P] beta;

    // GP parameter
    real logrho;
    real logalpha;
    real logsigma;

    // kernel choice
    // 1=Gaussian 2=Exponential
    int<lower=1, upper=2> kernel;
}

transformed data {
    // Parameters
    real rho = exp(logrho);
    real alpha = exp(logalpha);
    real sigma = exp(logsigma);
    
    // Prediction indexes
    array[N] int is_pt_valid = zeros_int_array(N);
    for(i in 1:Nvalid){
        is_pt_valid[ivalid[i]] = 1;
    }

    int Npred = N - Nvalid;
    array[Npred] int<lower=1, upper=N> ipred;
    int ii = 1;
    for(i in 1:N){
        if(is_pt_valid[i] == 0) {
            ipred[ii] = i;
            ii += 1;
        }    
    }

    // Covariance
    matrix[N, N] K;
    if(kernel==1) {
        K = gp_exp_quad_cov(w, alpha, rho);
    }   
    else if(kernel==2) {
        K = gp_exponential_cov(w, alpha, rho);
    }
    real sq_sigma = square(sigma);
    for(n in 1:N) {
        K[n, n] = K[n, n]+sq_sigma;
    }    

    // Multivariate normal cofactors
    matrix[Nvalid, Nvalid] K11 = K[ivalid, ivalid];
    matrix[Npred, Nvalid] K21 = K[ipred, ivalid];
    matrix[Npred, Npred] K22 = K[ipred, ipred];

    matrix[Npred, Nvalid] Kfactor = K21 / K11;
    matrix[Npred, Npred] Kpred = K22 - Kfactor * K21';

    // Mean
    vector[Npred] mu_pred = x[ipred] * beta 
        + Kfactor * (y[ivalid] - x[ivalid] * beta);
}

parameters {
    vector[Npred] yhat;
}

model {
    yhat ~ multi_normal(mu_pred, Kpred);
}


