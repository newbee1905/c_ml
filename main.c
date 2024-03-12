#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "nn.h"

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
		forward(params, X_train, y_hat, train_count, features_sz);
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
	forward(params, X_train, y_hat, train_count, features_sz);

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
	forward(params, X_test, y_test, test_count, features_sz);

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
