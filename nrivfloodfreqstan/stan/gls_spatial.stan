
data {
    int<lower=1> N;        // Number of data points
    int<lower=0> P;        // Number of predictors
    matrix[N, P] x;        // predictors
    array[N] vector[2] w;  // spatial coordinates
    vector[N] y;           // observed predictands

   // kernel choice
   // 1=Gaussian 2=Exponential
   int<lower=1, upper=2> kernel;
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
   real<lower=0> rho;
   real<lower=0> alpha;
   real<lower=0> sigma;
}

model {
    // Mean
    vector[N] mu = Q_ast*theta;

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

    // Prior
    rho ~ inv_gamma(5, 5);
    alpha ~ std_normal();
    sigma ~ std_normal();

    // Likelihood
    y ~ multi_normal_cholesky(mu, L);
}

generated quantities {
    vector[P] beta;
    beta = R_ast_inverse *  theta;
}

