#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

double train[][5] = {
	{1, 1, 1, 1, 4},
	{1, 2, 1, 1, 5},
	{1, 1, 2, 1, 5},
	{1, 1, 1, 2, 5},
	{1, 2, 2, 3, 8},
	{2, 3, 3, 2, 10},
};

double test[][4] = {
	{10, 10, 10, 10},
	{19, 21, 12, 15},
};

#define train_count (sizeof(train) / sizeof(train[0]))
#define test_count (sizeof(test) / sizeof(test[0]))

typedef struct {
	double* w;
	double* dw;
	double eps;
} loss_params_t;

typedef double (*loss_fn_t)(double *, double *, size_t);

loss_params_t* loss_params_init(size_t sz, double eps);
void loss_params_free(loss_params_t* params);
void loss_params_zero_weights(loss_params_t* params, size_t w_sz);
void loss_params_zero_weight(loss_params_t* params, size_t w_sz);

double rand_double(void);
double rmse(double* y_hat, double *y_true, size_t sz);

void predict(
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

int main() {
	srand(24);	

	double lr = 1e-3;
	size_t epoch_nums = 5e4;

	size_t features_sz = 4;

	double *X_train = malloc((train_count * features_sz + 1) * sizeof(*X_train));
	double *y_train = malloc(train_count * sizeof(*y_train));
	double *y_hat = malloc(train_count * sizeof(*y_train));
	double *y_hat_e = malloc(train_count * sizeof(*y_train));

	double *X_test = malloc((test_count * features_sz + 1) * sizeof(*X_test));
	double *y_test = malloc(test_count * sizeof(*y_test));

	loss_params_t* params = loss_params_init(train_count + 1, 0.0f);
	loss_params_zero_weight(params, features_sz + 1);

	for (size_t i = 0; i < train_count; ++i) {
		for (size_t j = 0; j < features_sz; ++j) {
			X_train[i * features_sz + j] = train[i][j];
		}
		y_train[i] = train[i][features_sz];
	}

	for (size_t i = 0; i < test_count; ++i) {
		for (size_t j = 0; j < features_sz; ++j) {
			X_test[i * features_sz + j] = test[i][j];
		}
	}

	loss_fn_t loss_fn = rmse;

	for (size_t epoch = 0; epoch < epoch_nums; ++epoch) {
		predict(params, X_train, y_hat, train_count, features_sz);
		double loss = (*loss_fn)(y_train, y_hat, train_count);
		printf("%lf\n", loss);
		backward(params, X_train, y_train, y_hat_e, train_count, features_sz, loss_fn, loss, lr);
	}

	printf("\n------------------------------\n\n");
	printf("WEIGHTS :\n");

	for (size_t j = 0; j <= features_sz; ++j) {
		printf("%lf, ", params->w[j]);
	}
	printf("\n");

	printf("\n------------------------------\n\n");
	printf("TRAIN:\n");
	predict(params, X_train, y_hat, train_count, features_sz);

	for (size_t i = 0; i < train_count; ++i) {
		for (size_t j = 0; j < features_sz; ++j) {
			printf("%lf, ", X_train[i * features_sz + j]);
		}
		printf("(%lf) ", y_train[i]);
		printf("-> %lf\n", y_hat[i]);
	}

	double loss = (*loss_fn)(y_train, y_hat, train_count);

	printf("LOSS %lf\n", loss);

	printf("\n------------------------------\n\n");

	printf("Validating: \n");
	predict(params, X_test, y_test, test_count, features_sz);

	for (size_t i = 0; i < test_count; ++i) {
		double sum = 0.0;
		for (size_t j = 0; j < features_sz; ++j) {
			sum += X_test[i * features_sz + j];
			printf("%lf, ", X_test[i * features_sz + j]);
		}
		printf("(%lf) ", sum);
		printf("-> %lf\n", y_test[i]);
	}

	free(X_train);
	free(y_train);
	free(y_hat);
	free(y_hat_e);
	free(X_test);
	free(y_test);
	loss_params_free(params);
	return 0;
}

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

void predict(
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
		predict(params, X_train, y_hat_e, train_sz, features_sz);
		double loss_e = (*loss_fn)(y_train, y_hat_e, train_sz);
		params->dw[i] = (loss_e - loss) / params->eps;
		params->w[i] -= params->eps;
	}

	for (size_t i = 0; i <= features_sz; ++i) {
		params->w[i] -= params->dw[i] * lr;
	}
}
