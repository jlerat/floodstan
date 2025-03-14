functions {

    #include logpearson3.stanfunctions

}


data {
  int N; // total number of values
  vector[N] y; // Data for first variable (ams streamflow)

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
  target += logpearson3_lpdf(y | ylocn, yscale, yshape1);
}


