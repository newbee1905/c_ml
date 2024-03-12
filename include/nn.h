#ifndef __NN_H__
#define __NN_H__

#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct {
	double* w;
	double* dw;
	double eps;
} loss_params_t;

typedef struct {
} layer_t;

typedef double (*loss_fn_t)(double *, double *, size_t);

loss_params_t* loss_params_init(size_t sz, double eps);
void loss_params_free(loss_params_t* params);
void loss_params_zero_weights(loss_params_t* params, size_t w_sz);
void loss_params_zero_weight(loss_params_t* params, size_t w_sz);

double rand_double(void);
double rmse(double* y_hat, double *y_true, size_t sz);

void forward(
	loss_params_t* params,
	double *X_train, double *y_hat,
	size_t train_sz, size_t features_sz
);

void backward(
	loss_params_t* params, 
	double *X_train, double *y_train, double *y_hat_e, 
	size_t train_sz, size_t features_sz,
	loss_fn_t loss_fn, double loss, double lr
);

#endif // __NN_H__
