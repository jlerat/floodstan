
/**
* Test functions

**/


functions {

    #include copula.stanfunctions

}


data {
  // Defines copula model
  // 1=Gumbel, 2=Clayton, 3=Gaussian
  int<lower=1, upper=3> copula; 

  int N; // total number of values
  matrix[N, 2] uv; // cdf data

  // Parameter for copula correlation
  real rho;
}  


generated quantities {
  real rho_check = rho;
    
  vector[N] luncens;
  vector[N] lcond;
  vector[N] lcens;

  vector[1] tmp1;
  matrix[1, 2] tmp2;

  for(i in 1:N){
    // Temporary vectors to match with function signature
    tmp1[1] = uv[i, 1];
    tmp2[1, 1] = uv[i, 1];
    tmp2[1, 2] = uv[i, 2];

    luncens[i] = copula_lpdf(tmp2| copula, rho);
    lcond[i] = copula_lpdf_conditional(tmp1, uv[i, 2], copula, rho);
    lcens[i] = copula_lcdf(tmp2| copula, rho);
  }


}


