data {
  int N;
  vector[N] x; 
  real alpha;
  real beta;
}  

generated quantities {
  vector[N] cdf;
  for(i in 1:N) {
      cdf[i] = gamma_cdf(x[i] | alpha, beta);
  }
}


