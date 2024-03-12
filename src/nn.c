#include "nn.h"

loss_params_t* loss_params_init(size_t sz, double eps) {
	loss_params_t* params = malloc(sizeof(loss_params_t));

	if (params == NULL) {
		return NULL;
	}

	params->w = malloc(sz * sizeof(double));
	if (params->w == NULL) {
		free(params);
		return NULL;
	}
	params->dw = malloc(sz * sizeof(double));
	if (params->dw == NULL) {
		free(params->w);
		free(params);
		return NULL;
	}

	if (eps > 0) {
		params->eps = eps;
	} else {
		params->eps = 1e-7;
	}

	return params;
}

void loss_params_free(loss_params_t* params) {
	if (params != NULL) {
		free(params->w);
		free(params->dw);
		free(params);
		params = NULL;
	}
}

void loss_params_zero_weights(loss_params_t* params, size_t w_sz) {
	for (size_t i = 0; i < w_sz; params->w[i++] = 0);
}

void loss_params_zero_weight(loss_params_t* params, size_t w_sz) {
	for (size_t i = 0; i < w_sz; params->w[i++] = rand_double());
}

double rand_double(void) {
	return (double)rand() / (double)RAND_MAX;
}

double rmse(double* y_hat, double *y_true, size_t sz) {
	double loss = 0;
	for (size_t i = 0; i < sz; ++i) {
		loss += (y_hat[i] - y_true[i]) * (y_hat[i] - y_true[i]);
	}

	return sqrt(loss / (sz * 1.0f));
}

void forward(
	loss_params_t* params,
	double *X_train, double *y_hat,
	size_t train_sz, size_t features_sz
) {

	for (size_t i = 0; i < train_sz; ++i) {
		y_hat[i] = params->w[0];
		for (size_t j = 0; j < features_sz; ++j) {
			y_hat[i] += X_train[i * features_sz + j] * params->w[j + 1];
		}
	}
}

void backward(
	loss_params_t* params, 
	double *X_train, double *y_train, double *y_hat_e, 
	size_t train_sz, size_t features_sz,
	loss_fn_t loss_fn, double loss, double lr
) {
	
	for (size_t i = 0; i <= features_sz; ++i) {
		params->w[i] += params->eps;
		forward(params, X_train, y_hat_e, train_sz, features_sz);
		double loss_e = (*loss_fn)(y_train, y_hat_e, train_sz);
		params->dw[i] = (loss_e - loss) / params->eps;
		params->w[i] -= params->eps;
	}

	for (size_t i = 0; i <= features_sz; ++i) {
		params->w[i] -= params->dw[i] * lr;
	}
}
