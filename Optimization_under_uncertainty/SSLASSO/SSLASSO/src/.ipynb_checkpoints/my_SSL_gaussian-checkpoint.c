#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

double crossprod(double *X, double *y, int n, int j);
double sum(double *x, int n);
double update_sigma2(double *r, int n);
int checkConvergence(double *beta, double *beta_old, double eps, int l, int p);
double SSL(double z, double beta, double lambda0, double lambda1, double theta, double v, int n, double delta, double sigma2);
double pstar(double x, double theta, double lambda1, double lambda0);
double lambdastar(double x, double theta, double lambda1, double lambda0);
double expectation_approx(double *beta, double a, double b, int p, int l);
double threshold(double theta, double sigma2, double lambda1, double lambda0, int n);


typedef struct {
    double *beta;
    double *loss;
    int *iter;
    double *thetas_export;
    double *sigmas_export;
} Result;

int get_length(const double *x) {
    int length = 0;

    // Iterate through the array until you encounter a sentinel value (e.g., NaN)
    while (!isnan(x[length])) {
        length++;
    }

    return length;
}


Result createResult(double *beta, double *loss, double *iter, double *thetas_export, double *sigmas_export) {
    Result result;

    // infer lengths
    int n1 = sizeof(beta) / sizeof(double);
    int n2 = sizeof(loss) / sizeof(double);
    int n3 = sizeof(iter) / sizeof(double);
    int n4 = sizeof(thetas_export) / sizeof(double);
    int n5 = sizeof(sigmas_export) / sizeof(double);

    // allocate memory
    result.beta = (double *)malloc(n1 * sizeof(double));
    result.loss = (double *)malloc(n2 * sizeof(double));
    result.iter = (int *)malloc(n3 * sizeof(int));
    result.thetas_export = (double *)malloc(n4 * sizeof(double));
    result.sigmas_export = (double *)malloc(n5 * sizeof(double));

    // Copy values from arguments to the result fields
    for (int i = 0; i < n1; i++) {
        result.beta[i] = beta[i];
    }
    for (int i = 0; i < n2; i++) {
        result.loss[i] = loss[i];
    }
    for (int i = 0; i < n3; i++) {
        result.iter[i] = iter[i];
    }
    for (int i = 0; i < n4; i++) {
        result.thetas_export[i] = thetas_export[i];
    }
    for (int i = 0; i < n5; i++) {
        result.sigmas_export[i] = sigmas_export[i];
    }

    return result;
}

void freeResult(Result result) {
    free(result.beta);
    free(result.loss);
    free(result.iter);
    free(result.thetas_export);
    free(result.sigmas_export);
}

// Memory handling, output formatting (Gaussian)

double cleanupG(double *a, double *r, int *e1, int *e2, double *z, double *beta, double *loss, double *iter, double *thetas_export, double *sigmas_export) {

  free(a);
  free(r);
  free(e1);
  free(e2);
  free(z);
    
  // Create the result container
  Result res = createResult(beta, loss, iter, thetas_export, sigmas_export);
  
  /*
  Free(a);
  Free(r);
  Free(e1);
  Free(e2);
  Free(z);

  SEXP res;
  PROTECT(res = allocVector(VECSXP, 5));
  SET_VECTOR_ELT(res, 0, beta);
  SET_VECTOR_ELT(res, 1, loss);
  SET_VECTOR_ELT(res, 2, iter);
  SET_VECTOR_ELT(res, 3, thetas_export);
  SET_VECTOR_ELT(res, 4, sigmas_export);

  UNPROTECT(1);
  */

  return(res);
}

// Gaussian loss
double gLoss(double *r, int n) {
  double l = 0;
  for (int i = 0; i < n; i++) l = l + pow(r[i], 2);
    return(l);
}

Result SSL_gaussian(const double *X, const double *y, const char *penalty, const double *variance,
                  double lambda1, const double *lambda0s, double theta, double sigma, double min_sigma2,
                  double a, double b, double eps, int max_iter, int counter) {

  // Declarations
  int n = get_length(y);
  int p = get_length(X)/n;
  int L = get_length(lambda0s);

  //double *X = REAL(X_);
  //double *y = REAL(y_);

  /*
  R LEGACY
  const char *penalty = CHAR(STRING_ELT(penalty_, 0));
  const char *variance = CHAR(STRING_ELT(variance_, 0));
  */
  
  /*
  R LEGACY
  double lambda1 = REAL(lambda1_)[0];
  double *lambda0s = REAL(lambda0s_);
  double lambda0;
  double theta = REAL(theta_)[0];
  double sigma2 = pow(REAL(sigma_)[0], 2);
  double sigma2_init = pow(REAL(sigma_)[0], 2);
  double min_sigma2 = REAL(min_sigma2_)[0];
  double aa = REAL(a_)[0];
  double bb = REAL(b_)[0];
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  int count_max = INTEGER(counter_)[0];
  */
    
  // new
  double lambda0;
  double sigma2 = pow(sigma, 2);
  double sigma2_init = pow(sigma, 2);

  // create containers for R output
  // SEXP res, beta, loss, iter, thetas_export, sigmas_export;
    

  //PROTECT(beta = allocVector(REALSXP, L*p));
  //double *b = REAL(beta);
  double *beta = (double *)malloc(L * p * sizeof(double));
  for (int j=0; j< (L*p); j++) {
    beta[j] = 0;
  }

  // R
  //PROTECT(thetas_export = allocVector(REALSXP, L));
  //double *thetas = REAL(thetas_export);
  double *thetas = (double *)malloc(L * sizeof(double));
    
  //PROTECT(sigmas_export = allocVector(REALSXP, L));
  //double *sigmas = REAL(sigmas_export);
  double *sigmas = (double *)malloc(L * sizeof(double));
  double *loss   = (double *)malloc(L * sizeof(double));

  for(int l = 0; l < L; l++) {
    thetas[l] = NAN;
    sigmas[l] = NAN;
  }

  // R legacy
  //PROTECT(loss = allocVector(REALSXP, L));
  //PROTECT(iter = allocVector(INTSXP, L));

  //for (int i = 0; i < L; i++) { 
  //  INTEGER(iter)[i] = 0;
  //}
  int *iter = (int *)malloc(L * sizeof(int));
  for (int i = 0; i < L; i++) {
      iter[i] = 0;
  }

  double delta = 0;

  // Beta from previous iteration
  // R LEGACY 
  // double *a = Calloc(p, double); 
  double *a = (double *)malloc(p * sizeof(double));

  for (int j = 0; j < p; j++) {
    a[j] = 0;
  }

  // Residuals
  //double *r = Calloc(n, double);
  double *r = (double *)malloc(n * sizeof(double));
  for (int i=0; i<n; i++) {
    r[i] = y[i];
  }

  // double *z = Calloc(p, double);
  double *z = (double *)malloc(p * sizeof(double));

  for (int j=0; j<p; j++) {
    z[j] = crossprod(X, r, n, j);
  }

  // Index of an active set
  //int *e1 = Calloc(p, int); 
  int *e1 = (int *)malloc(p * sizeof(int));

  for (int j=0; j<p; j++) {
    e1[j] = 0;
  }

  // Index of an eligible set from the strong rule
  //int *e2 = Calloc(p, int);
  int *e2 = (int *)malloc(p * sizeof(int));

  for (int j=0; j<p; j++) {
    e2[j] = 0;
  }

  // Thresholds for the strong screening rule
  //double *thresholds = Calloc(L, double); 
  double *thresholds = (double *)malloc(L * sizeof(double));

  double cutoff;
  int converged = 0;
  int counter = 0;
  int violations = 0;
  int estimate_sigma = 0;

  // Regularization Path

  for (int l = 0; l < L; l++) {

    // Might need to handle interuptions
    // R_CheckUserInterrupt();

    lambda0 = lambda0s[l];

    if (l != 0) {

      if (strcmp(penalty, "adaptive") == 0){
        theta = expectation_approx(b, aa, bb, p, l - 1);
      }
      if (strcmp(variance, "unknown") == 0) {
        if (iter[l - 1] < 100) {
          estimate_sigma = 1;
          sigma2 = update_sigma2(r, n);
          if(sigma2 < min_sigma2) {
            sigma2 = sigma2_init;
            estimate_sigma = 0;
          }
        } else {
          estimate_sigma = 0;
          if(iter[l - 1] == max_iter) {
            sigma2 = sigma2_init;
          }
        }

      }

      thresholds[l] = threshold(theta, sigma2, lambda1, lambda0, n);

      // Determine eligible set

      cutoff = thresholds[l];

      for (int j=0; j < p; j++) {
        if (fabs(z[j]) > cutoff) {
          e2[j] = 1;
        }
      }

    } else {

      thresholds[l] = threshold(theta, sigma2, lambda1, lambda0, n);

      cutoff = thresholds[l];

      for (int j=0; j<p; j++) {
        if (fabs(z[j]) > cutoff) {
          e2[j] = 1;
        }
      }

    }

    delta = thresholds[l];

    while (iter[l] < max_iter) { 
      while (iter[l] < max_iter) {
        while (iter[l] < max_iter) {

            // Solve over the active set

          iter[l]++;

          for (int j=0; j<p; j++) {


            if (e1[j]) {

                // Update residuals zj
              z[j] = crossprod(X, r, n, j) + n * a[j];

                // Update beta_j

              b[l*p+j] = SSL(z[j], a[j], lambda0, lambda1, theta, 1, n, delta, sigma2);

                // Update r

              double shift = b[l*p+j] - a[j];

              if (shift !=0) {
                for (int i=0;i<n;i++) {
                  r[i] -= shift*X[j*n+i];
                }
              }
              counter++;
            }

              // Update theta every count_max iterations

            if (counter == count_max){
              if(strcmp(penalty, "adaptive")==0) {
                theta = expectation_approx(b, aa, bb,p,l);
                delta = threshold(theta, sigma2, lambda1, lambda0, n); 
              }

              if(strcmp(variance, "unknown") == 0 && estimate_sigma) {
                sigma2 = update_sigma2(r, n);
                if(sigma2 < min_sigma2) {
                  sigma2 = sigma2_init;
                }
              }

              counter = 0;
            }

          }

            // Check for convergence over the active set

          converged = checkConvergence(b, a, eps, l, p);

          for (int j = 0; j < p; j++) {
            a[j] = b[l * p + j];
          }

          if (converged) {
            break;
          }
        }

          // Scan for violations in strong set

        violations = 0;
        counter = 0;


        for (int j=0; j < p; j++) {

          if (e1[j] == 0 && e2[j] == 1) {

            z[j] = crossprod(X, r, n, j) + n * a[j];

            // Update beta_j

            b[l*p+j] = SSL(z[j], a[j], lambda0, lambda1, theta, 1, n, delta, sigma2);

            // If something enters the eligible set, update eligible set & residuals

            if (b[l*p+j] !=0) {

              e1[j] = e2[j] = 1;

              for (int i=0; i<n; i++) {
                r[i] -= b[l*p+j]*X[j*n+i];
              }

              a[j] = b[l*p+j];

              violations++;
              counter++;
            }
          }

          if (counter == count_max) {

            if(strcmp(penalty, "adaptive") == 0) {
              theta = expectation_approx(b, aa, bb,p,l);
              delta = threshold(theta, sigma2, lambda1, lambda0, n); 
            }

            if(strcmp(variance, "unknown") == 0 && estimate_sigma) {
              sigma2 = update_sigma2(r, n);
              if(sigma2 < min_sigma2) {
                sigma2 = sigma2_init;
              }
            }

            counter=0;
          }

        }

        if (violations==0) break;
      }

      // Scan for violations in rest

      int violations = 0;

      counter=0;

      for (int j=0; j<p; j++) {

        if (e2[j] == 0) {

          z[j] = crossprod(X, r, n, j) + n * a[j];

          // Update beta_j

          b[l*p+j] = SSL(z[j], a[j], lambda0, lambda1, theta, 1, n, delta, sigma2);

          // If something enters the eligible set, update eligible set & residuals

          if (b[l*p+j] !=0) {

            e1[j] = e2[j] = 1;

            for (int i=0; i<n; i++) {
              r[i] -= b[l*p + j] * X[j*n + i];
            }

            a[j] = b[l*p+j];

            violations++;
            counter++;

          }
        }

        if (counter == count_max){

          if(strcmp(penalty, "adaptive") == 0) {
            theta = expectation_approx(b, aa, bb,p,l);
            delta = threshold(theta, sigma2, lambda1, lambda0, n); 
          }
          if(strcmp(variance, "unknown") == 0 && estimate_sigma) {
            sigma2 = update_sigma2(r, n);
            if(sigma2 < min_sigma2) {
              sigma2 = sigma2_init;
            }
          }
          counter = 0;
        }
      }


      if (violations == 0) {
        //REAL(loss)[l] = gLoss(r, n);
        loss[l] = gLoss(r, n);
        thetas[l] = theta;
        sigmas[l] = sqrt(sigma2);

        break;
      }
    }
  }

  res = cleanupG(a, r, e1, e2, z, beta, loss, iter, thetas_export, sigmas_export);
  //UNPROTECT(5);

  return(res);
}
