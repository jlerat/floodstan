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

  vector[100] u = linspaced_vector(100, -1e-10, 1e-10);
  vector[100] tu;
  vector[100] lpdf1;
  vector[100] lpdf2;

  for(i in 1:100){
      tu[i] = trans(u[i]); 
      lpdf1[i] = gamma_lpdf(tu[i] | 1.01, 1.);
      lpdf2[i] = gamma_lpdf(tu[i] | 1.1, 1.);
  }


  for(i in 1:N) {
      yi = y[i];
      yv[1] = yi;
      ydata[i] = yi;
      luncens[i] = logpearson3_lpdf(yv | ylocn, yscale, yshape1);
      cens[i] = logpearson3_cdf(yi | ylocn, yscale, yshape1);
      lcens[i] = logpearson3_lcdf(yi | ylocn, yscale, yshape1);
  }
}


