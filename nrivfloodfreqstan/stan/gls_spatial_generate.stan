
data {
    int<lower=1> N;        // Number of data points
    int<lower=0> P;        // Number of predictors
    matrix[N, P] x;        // predictors
    array[N] vector[2] w;  // spatial coordinates
   
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

    // Mean
    vector[N] mu = x*beta;

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
    matrix[N, N] L = cholesky_decompose(K);
}

parameters {
    vector[N] eta;
}

model {
    eta ~ std_normal();
}

generated quantities {
    vector[N] y;
    y = mu+L*eta;
}


