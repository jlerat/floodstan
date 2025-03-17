functions {

    #include logpearson3.stanfunctions

}


data {
  int N; // total number of values
  vector[N] y; // Data for first variable (ams streamflow)

  vector[2] ylocn_prior;
  vector[2] ylogscale_prior;
  vector[2] yshape1_prior;
}

parameters {
  real ylocn; 
  real ylogscale;
  real yshape1;
}  

transformed parameters {
  real yscale = exp(ylogscale);
}


model {
  ylocn ~ normal(ylocn_prior[1], ylocn_prior[2]) T[locn_lower, locn_upper];
  ylogscale ~ normal(ylogscale_prior[1], ylogscale_prior[2]) T[logscale_lower, logscale_upper];
  yshape1 ~ normal(yshape1_prior[1], yshape1_prior[2]) T[shape1_lower, shape1_upper];

  target += logpearson3_lpdf(y | ylocn, yscale, yshape1);
}


