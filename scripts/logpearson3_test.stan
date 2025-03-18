functions {
    
    #include logpearson3.stanfunctions

}

data {
  real ylocn; 
  real ylogscale;
  real yshape1;
  int N;
  real ylower;
  real yupper;
}  

transformed data {
  real yscale = exp(ylogscale);
  real alpha = 4. / yshape1^2;
  real beta = 2. / yshape1 / yscale;
  real tau = ylocn - alpha / beta;
  vector[N] y = linspaced_vector(N, ylower, yupper);
}

generated quantities {
  real alpha_copy = alpha;
  real beta_copy = beta;
  real abs_beta = abs(beta);
  real tau_copy = tau;

  vector[N] ydata;
  vector[N] lpdf;
  vector[N] cdf;
  vector[N] lcdf;
  //vector[N] trans_cdf;
  //vector[N] trans_pdf;
  vector[1] yv;
  vector[N] u;
  real yi;
  //real lin_lcdf_trans = logpearson3_lcdf_linear_transition(abs(yshape1));
  //real lin_lpdf_trans = logpearson3_lpdf_linear_transition();

  //real f0 = gamma_lcdf(lin_lcdf_trans | alpha, 1.);
  //real ldf0 = gamma_lpdf(lin_lcdf_trans | alpha, 1.) - f0;
  //real df0 = exp(ldf0);

  for(i in 1:N) {
      yi = y[i];
      yv[1] = yi;
      ydata[i] = yi;
      lpdf[i] = logpearson3_lpdf(yv | ylocn, yscale, yshape1);
      cdf[i] = logpearson3_cdf(yi | ylocn, yscale, yshape1);
      lcdf[i] = logpearson3_lcdf(yi | ylocn, yscale, yshape1);

      //u[i] = sign(yshape1) * (log(yi)  - tau);
      //trans_pdf[i] = u[i] > lin_lpdf_trans ? 1. : 0.;
      //trans_cdf[i] = u[i] > lin_lcdf_trans ? 1. : 0.;
  }
}


