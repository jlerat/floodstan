functions {
    
    #include logpearson3.stanfunctions

}

data {
  real ylocn; 
  real ylogscale;
  real yshape1;
  int N;
  vector[N] y; 
}  

transformed data {
  real yscale = exp(ylogscale);
  real alpha = 4. / yshape1^2;
  real beta = 2. / yshape1 / yscale;
  real tau = ylocn - alpha / beta;
}

generated quantities {
  real alpha_copy = alpha;
  real beta_copy = beta;
  real abs_beta = abs(beta);
  real tau_copy = tau;

  vector[N] ydata;
  vector[N] luncens;
  vector[N] cens;
  vector[N] lcens;
  vector[1] yv;
  real yi;

  for(i in 1:N) {
      yi = y[i];
      yv[1] = yi;
      ydata[i] = yi;
      luncens[i] = logpearson3_lpdf(yv | ylocn, yscale, yshape1);
      cens[i] = logpearson3_cdf(yi | ylocn, yscale, yshape1);
      lcens[i] = logpearson3_lcdf(yi | ylocn, yscale, yshape1);
  }
}


