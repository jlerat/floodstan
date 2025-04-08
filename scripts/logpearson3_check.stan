functions {
    
    #include logpearson3.stanfunctions

}

data {
  int N;
  vector[N] y; 

  // Indexing
  // .. number of values - 9 cases (only 2 used in univariate fitting)
  array[3, 3] int<lower=0, upper=N> Ncases;

  // 2 cases obs/censored
  array[Ncases[1,1]] int<lower=1, upper=N> i11;
  array[Ncases[2,1]] int<lower=1, upper=N> i21;

  real ylocn; 
  real ylogscale;
  real yshape1;

}  

generated quantities {
  real ylocn_check = ylocn;
  real ylogscale_check = ylogscale;
  real yshape1_check = yshape1;

  real yscale = exp(ylogscale);
  real alpha = 4. / yshape1^2;
  real beta = 2. / yshape1 / yscale;
  real tau = ylocn - alpha / beta;

  vector[N] ydata;
  vector[N] luncens;
  vector[N] cens;
  vector[N] lcens;
  vector[1] yv;
  real yi;

  for(i in i11) {
      yi = y[i];
      yv[1] = yi;
      ydata[i] = yi;
      luncens[i] = logpearson3_lpdf(yv | ylocn, yscale, yshape1);
      cens[i] = logpearson3_cdf(yi | ylocn, yscale, yshape1);
      lcens[i] = logpearson3_lcdf(yi | ylocn, yscale, yshape1);
  }
}


