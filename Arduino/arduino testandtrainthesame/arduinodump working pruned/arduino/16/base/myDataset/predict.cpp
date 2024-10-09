#include <Arduino.h>

#include "config.h"
#include "predict.h"
#include "library.h"
#include "model.h"

using namespace model;

int predict() {
	int16_t tmp5[180][1];
	MYINT tmp6;
	MYINT tmp7;
	int16_t tmp9[8][1];
	int16_t tmp10[8][1];
	int16_t tmp12[5][1];
	int16_t tmp13[5][1];
	MYINT tmp14;



	// reshape(X, (180, 1), (1, 2
	tmp6 = 0;
	tmp7 = 0;
	for (int i0 = 0; (i0 < 180); i0++) {
		char scratch[0];
		for (int i1 = 0; (i1 < 1); i1++) {
			char scratch[0];
			tmp5[tmp6][tmp7] = getIntFeature(i0 * 1 + i1 * 1);
			tmp7 = (tmp7 + 1);
			if ((tmp7 == 1)) {
				tmp7 = 0;
				tmp6 = (tmp6 + 1);
			}
		}
	}

	// W1 * XX
	{
		int32_t tmp8[180];
		MatMulCN<int16_t, int16_t, int32_t, int16_t>((int16_t*)&W1[0][0], (int16_t*)&tmp5[0][0], (int16_t*)&tmp9[0][0], tmp8, 8, 180, 1, 1, 1, 8, 0, 1);
	}

	// tmp9 + Bias1
	{
		MatAddNC<int16_t, int16_t, int16_t, int16_t>((int16_t*)&tmp9[0][0], (int16_t*)&Bias1[0][0], (int16_t*)&tmp10[0][0], 8, 1, 1, 32, 1, 1);
	}

	// relu(tmp10)
	{
		Relu2D((int16_t*)&tmp10[0][0], 8, 1);
	}

	// W2 * tmp10
	{
		int32_t tmp11[8];
		MatMulCN<int16_t, int16_t, int32_t, int16_t>((int16_t*)&W2[0][0], (int16_t*)&tmp10[0][0], (int16_t*)&tmp12[0][0], tmp11, 5, 8, 1, 64, 64, 3, 0, 1);
	}

	// tmp12 + Bias2
	{
		MatAddNC<int16_t, int16_t, int16_t, int16_t>((int16_t*)&tmp12[0][0], (int16_t*)&Bias2[0][0], (int16_t*)&tmp13[0][0], 5, 1, 1, 64, 1, 1);
	}

	// argmax(tmp13)
	{
		ArgMax<int16_t>((int16_t*)&tmp13[0][0], 5, 1, &tmp14);
	}

	return tmp14;
}
