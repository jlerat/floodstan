data {
  int N;
  vector[N] x; 
  real a;
}  

generated quantities {
  vector[N] cdf;
  for(i in 1:N) {
      cdf[i] = gamma_cdf(x[i] | a, 1.);
  }
}


