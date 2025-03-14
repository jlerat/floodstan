functions {
    
    #include logpearson3.stanfunctions

}


data {
  real ylocn; 
  real ylogscale;
  real yshape1;
}  

transformed data {
  int N = 500;
  real yscale = exp(ylogscale);
  real alpha = 4. / yshape1^2;
  real beta = 2. / yshape1 / yscale;
  real tau = ylocn - alpha / beta;
  
  real ylower = exp(tau) - 0.1;
  real yupper = exp(tau) + 0.1;
  vector[N] y = linspaced_vector(N, ylower, yupper);

}

generated quantities {
  real alpha_copy = alpha;
  real beta_copy = beta;
  real tau_copy = tau;

  vector[N] ydata;
  vector[N] lpdf;
  //vector[N] cdf;
  vector[N] lcdf;
  vector[1] yv;

  for(i in 1:N) {
      yv[1] = y[i];
      ydata[i] = y[i];
      lpdf[i] = logpearson3_lpdf(yv | ylocn, yscale, yshape1);
      //cdf[i] = logpearson3_cdf(y[i] | ylocn, yscale, yshape1);
      lcdf[i] = logpearson3_lcdf(y[i] | ylocn, yscale, yshape1);
  }
}


