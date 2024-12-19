
data {
    int<lower=1> N;        // Number of data points
    int<lower=0> P;        // Number of predictors
    int<lower=0> Nvalid;   // Number of valid observations

    matrix[N, P] x;        // predictors
    array[N] vector[2] w;  // spatial coordinates
    vector[N] y;           // observed predictands

    array[Nvalid] int<lower=1, upper=N> ivalid;

    // kernel choice
    // 1=Gaussian 2=Exponential
    int<lower=1, upper=2> kernel;

    // Priors
    real<lower=-10> logrho_lower;
    real<lower=logrho_lower, upper=20> logrho_upper;
    vector[2] logrho_prior;
    vector[2] logalpha_prior;
    vector[2] logsigma_prior;
    matrix[2, P] theta_prior;
}  

transformed data {
  // QR reparameterization (
  // ee https://mc-stan.org/docs/stan-users-guide/QR-reparameterization.html 
  matrix[N, P] Q_ast;
  matrix[P, P] R_ast;
  matrix[P, P] R_ast_inverse;

  // thin and scale the QR decomposition
  Q_ast = qr_thin_Q(x) * sqrt(N - 1);
  R_ast = qr_thin_R(x) / sqrt(N - 1);
  R_ast_inverse = inverse(R_ast);
}

parameters {
   // Mean parameters
   vector[P] theta;        // Predictor coefficients on Q_ast

   // GP parameter
   real logrho;
   real logalpha;
   real logsigma;
}

transformed parameters {
    real rho = exp(logrho);
    real alpha = exp(logalpha);
    real sigma = exp(logsigma);

    // Mean vector
    vector[N] mu0 = Q_ast*theta;
}

model {
    // See https://mc-stan.org/docs/stan-users-guide/gaussian-processes.html#fit-gp.section

    // Subset of mean vector
    vector[Nvalid] mu;
    for(n in 1:Nvalid) {
        mu[n] = mu0[ivalid[n]];
    }

    // Covariance
    matrix[Nvalid, Nvalid] K;
    if(kernel==1) {
        K = gp_exp_quad_cov(w[ivalid], alpha, rho);
    }   
    else if(kernel==2) {
        K = gp_exponential_cov(w[ivalid], alpha, rho);
    }
 
    real sq_sigma = square(sigma);
    for(n in 1:Nvalid) {
        K[n, n] = K[n, n]+sq_sigma;
    }    
    matrix[Nvalid, Nvalid] L = cholesky_decompose(K);

    // Prior
    logrho ~ normal(logrho_prior[1], logrho_prior[2]) T[logrho_lower, logrho_upper]; 
    logalpha ~ normal(logalpha_prior[1], logalpha_prior[2]);
    logsigma ~ normal(logsigma_prior[1], logsigma_prior[2]);

    for(p in 1:P) {
        theta[p] ~ normal(theta_prior[1, p], theta_prior[2, p]);
    }

    // Likelihood
    y[ivalid] ~ multi_normal_cholesky(mu, L);
}

generated quantities {
    vector[P] beta;
    beta = R_ast_inverse *  theta;
}

