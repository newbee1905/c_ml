#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

double train[][2] = {
	{0, 0},
	{1, 2},
	{2, 4},
	{3, 6},
	{4, 8},
	{5, 10},
	{6, 12},
	{7, 14},
};

#define train_count (sizeof(train) / sizeof(train[0]))

double rand_double(void);
double rmse(double* y_hat, double *y_true, size_t sz);

int main() {
	srand(24);	

	double w = rand_double() * 10.0f;
	double lr = 1e-3;
	double e = 1e-7;

	double *X_train = malloc(train_count * sizeof(*X_train));
	double *y_train = malloc(train_count * sizeof(*y_train));
	double *y_hat = malloc(train_count * sizeof(*y_train));
	double *y_hat_e = malloc(train_count * sizeof(*y_train));

	for (size_t i = 0; i < train_count; ++i) {
		X_train[i] = train[i][0];
		y_train[i] = train[i][1];
	}

	for (size_t epoch = 0; epoch < 5000; ++epoch) {
		for (size_t i = 0; i < train_count; ++i) {
			y_hat[i] = X_train[i] * w;
			y_hat_e[i] = X_train[i] * (w + e);
		}
		double loss = rmse(y_hat, y_train, train_count);
		double loss_e = rmse(y_hat_e, y_train, train_count);
		double diff_w = (loss_e - loss) / e;
		w -= lr * diff_w;
		printf("rmse = %lf, w = %lf\n", loss, w);
	}

	for (size_t i = 0; i < train_count; ++i) {
		y_hat[i] = X_train[i] * w;
		printf("X_train: %lf, y_train: %lf, y_hat: %lf\n", X_train[i], y_train[i], y_hat[i]);
	}
	double loss = rmse(y_hat, y_train, train_count);

	printf("------------------------------\n");
	printf("RMSE %lf\n", loss);

	free(X_train);
	free(y_train);
	free(y_hat);
	free(y_hat_e);
	return 0;
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
