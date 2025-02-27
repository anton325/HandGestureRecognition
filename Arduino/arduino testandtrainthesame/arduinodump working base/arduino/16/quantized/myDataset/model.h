#pragma once

namespace model {

const PROGMEM float X[180] = {
	0.677734f, 0.693359f, 0.712891f, 0.708008f, 0.669922f, 0.711914f, 0.714844f, 0.675781f, 0.662109f, 0.679688f, 0.695364f, 0.714895f, 0.710835f, 0.674548f, 0.714638f, 0.713353f, 0.673674f, 0.659334f, 0.679482f, 0.696083f, 0.715512f, 0.704975f, 0.666427f, 0.708984f, 0.611174f, 0.580952f, 0.576377f, 0.673263f, 0.690944f, 0.711349f, 0.628032f, 0.596012f, 0.664474f, 0.233296f, 0.227179f, 0.253803f, 0.593287f, 0.628238f, 0.674342f, 0.328690f, 0.306332f, 0.418586f, 0.035516f, 0.023643f, 0.025391f, 0.294665f, 0.355315f, 0.435341f, 0.053968f, 0.028269f, 0.044922f, 0.017013f, 0.008995f, 0.009714f, 0.038292f, 0.046618f, 0.052323f, 0.024568f, 0.010177f, 0.015419f, 0.012079f, 0.006219f, 0.006219f, 0.015419f, 0.016036f, 0.014700f, 0.017116f, 0.006116f, 0.009663f, 0.010023f, 0.004523f, 0.004523f, 0.010485f, 0.010485f, 0.008532f, 0.013826f, 0.004472f, 0.007401f, 0.008789f, 0.003906f, 0.003906f, 0.008326f, 0.007864f, 0.006373f, 0.011770f, 0.003444f, 0.006373f, 0.009714f, 0.006682f, 0.002981f, 0.007299f, 0.006322f, 0.005345f, 0.011256f, 0.007042f, 0.005859f, 0.009714f, 0.007710f, 0.001953f, 0.006836f, 0.012079f, 0.004883f, 0.011719f, 0.009046f, 0.005294f, 0.044973f, 0.005859f, 0.001953f, 0.006836f, 0.017218f, 0.004266f, 0.016653f, 0.005962f, 0.004266f, 0.327868f, 0.052734f, 0.003187f, 0.007504f, 0.011565f, 0.003238f, 0.133121f, 0.015574f, 0.003906f, 0.599147f, 0.299239f, 0.037315f, 0.049548f, 0.010948f, 0.002930f, 0.455387f, 0.136359f, 0.019737f, 0.694130f, 0.555099f, 0.283717f, 0.332751f, 0.097296f, 0.006785f, 0.653526f, 0.444490f, 0.157227f, 0.711092f, 0.650442f, 0.541992f, 0.586863f, 0.418637f, 0.063734f, 0.702046f, 0.625000f, 0.511462f, 0.716026f, 0.674291f, 0.643966f, 0.664782f, 0.630140f, 0.386153f, 0.707802f, 0.664731f, 0.677889f, 0.714176f, 0.674137f, 0.657792f, 0.673623f, 0.682669f, 0.640265f, 0.707083f, 0.668843f, 0.708008f, 0.716643f, 0.677529f, 0.662829f, 0.679688f, 0.695312f, 0.709961f, 0.709961f, 0.672852f, 0.713867f, 0.716797f, 0.676758f, 0.662109f, 
};

const PROGMEM int8_t Xint[180] = {
	86, 88, 91, 90, 85, 91, 91, 86, 84, 87, 89, 91, 90, 86, 91, 91, 86, 84, 86, 89, 91, 90, 85, 90, 78, 74, 73, 86, 88, 91, 80, 76, 85, 29, 29, 32, 75, 80, 86, 42, 39, 53, 4, 3, 3, 37, 45, 55, 6, 3, 5, 2, 1, 1, 4, 5, 6, 3, 1, 1, 1, 0, 0, 1, 2, 1, 2, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 5, 0, 0, 0, 2, 0, 2, 0, 0, 41, 6, 0, 0, 1, 0, 17, 1, 0, 76, 38, 4, 6, 1, 0, 58, 17, 2, 88, 71, 36, 42, 12, 0, 83, 56, 20, 91, 83, 69, 75, 53, 8, 89, 79, 65, 91, 86, 82, 85, 80, 49, 90, 85, 86, 91, 86, 84, 86, 87, 81, 90, 85, 90, 91, 86, 84, 87, 89, 90, 90, 86, 91, 91, 86, 84, 
};

const int scaleOfX = -7;

const int Y = 0;

const PROGMEM int8_t W1[8][180] = {
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -25, 0, 0, 0, 0, 0, 0, 0, 0, -31, 0, 0, 0, 18, 0, 0, 27, 0, -39, 27, 0, 0, 27, 0, 0, 18, 0, -57, 36, 0, 0, 27, 0, 0, 27, 0, -57, 36, 0, 0, 27, 0, 0, 36, 0, -57, 36, 0, 0, 36, 0, 0, 27, 0, -47, 36, 27, 0, 36, 0, 0, 27, 0, -39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -25, 0, 0, -39, 0, 0, -39, -25, 27, -39, 0, 0, -39, 0, 0, -39, -25, 47, -47, 0, 0, -31, 0, 18, -25, 0, 58, -47, 0, 0, -31, 0, 27, 0, 0, 72, -39, 0, 0, -31, 0, 27, 0, 0, 72, 0, 0, 0, 0, 0, 18, 0, 0, 58, 0, 0, 0, 0, 0, 18, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 27, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 27, 0, 0, 0, 0, 0, 0, 27, 27, 0, 0, 0, 0, 0, 0, 0, 36, 36, 0, 18, 0, 0, 0, 0, -31, 47, 0, 0, 18, 0, 0, 0, -31, -67, 58, 0, 0, 0, 0, 0, -39, -39, -67, 58, 36, 0, 0, 0, -11, -39, -47, -67, 58, 36, 0, 0, 0, 0, 0, -31, -57, 58, 27, 36, 0, 0, 0, 0, -25, -39, 47, 0, 0, 0, 0, 0, 0, 0, -39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -57, -39, 0, -25, 0, 0, 36, 47, 58, -79, -57, -11, -39, 0, 0, 0, 47, 72, -67, -57, -31, -31, 0, 0, 0, 58, 87, -67, -47, 0, -25, 0, 0, 0, 47, 72, -57, 0, 0, -31, 0, 0, 0, 36, 72, -39, 0, 0, 0, 0, 0, 0, 0, 58, 0, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 36, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 36, 0, 36, 0, 0, 0, 0, 18, 36, 58, 0, 0, 0, -67, 0, 18, 0, 36, 58, -47, 0, 0, -98, -47, 0, 0, 36, 72, -47, 0, 0, -98, -47, 0, 0, 0, 72, -25, 0, 0, -79, -11, 0, 0, 0, 87, -11, 0, 0, -47, 18, 0, 0, 0, 0, 0, 0, 0, -25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -31, 0, 0, 0, 0, 0, 0, 0, 0, -47, -25, 27, 0, 0, 36, 58, -11, 0, -47, -47, 27, 0, -39, 58, 58, -25, 0, -47, -47, 27, 0, -47, 58, 72, -39, 0, -47, -57, 27, 0, -47, 72, 58, -47, 18, -47, -39, 36, 0, -47, 58, 47, -57, 0, 0, 0, 27, 0, 0, 36, 47, -47, 0, 0, 0, 0, 0, 0, 18, 36, -39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	47, 36, 0, 36, 0, 0, 0, 0, 27, 27, 47, 0, 36, 0, 0, 0, -25, 0, 0, 72, 0, 0, -31, 0, 0, -79, 0, 0, 0, 0, 0, -31, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 0, 18, 0, 58, -47, 47, 0, 47, 18, 0, 18, 0, 47, -79, 0, 0, -11, 18, 0, 18, 36, 58, -47, 0, 0, -39, 0, 0, 0, 0, 47, -39, 0, 0, 0, 0, 0, 36, 0, 58, -39, 0, 0, 0, 0, -31, 58, 0, 58, 0, 0, 0, 0, 18, 0, 58, 0, 47, -39, 0, -47, -25, 27, 0, 47, 18, 72, -31, 27, -31, -31, 0, 0, 36, 0, 36, -79, 27, -47, -39, 0, 0, 58, 18, 27, -98, 0, -67, -79, 0, 0, 0, 0, 58, -67, 0, -31, -25, -39, -11, 0, 0, 18, 0, 0, -39, 0, -39, -47, 0, 0, 0, 0, -31, -57, 0, -39, -31, 0, 0, 0, -39, 0, -39, 0, 0, -31, 0, 27, 27, 0, 0, 0, 0, 0, -31, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 27, 0, 0, 0, -31, 0, 0, 0, 36, 27, 0, 0, -31, -39, 0, 0, 0, 58, 36, 0, 0, -39, -39, 27, 0, -31, 72, 58, 0, 0, -47, -39, 27, 0, -31, 72, 72, 0, -31, -57, -57, 0, 0, -25, 72, 72, 0, -25, -57, -57, 0, 0, -25, 58, 58, 0, -11, -31, -47, 0, 0, 0, 36, 47, 0, 0, 0, -25, 0, 0, -25, 0, 0, 0, 18, 0, 0, 0, 0, -31, -39, 0, -25, 18, 0, 0, 0, 0, 0, -67, -39, -25, 18, 27, 18, 0, 0, 0, -67, -47, -11, 18, 27, 18, -25, 0, 0, -67, -47, 0, 18, 27, 36, -25, 0, 0, -67, -57, 0, 27, 18, 36, 0, 0, 0, -47, -31, 0, 36, 0, 0, 0, 0, 0, -25, 0, 0, 36, 27, 0, 0, 27, 0, 0, 0, 27, 36, 27, 27, 0, 27, 0, 0, 0, 0, 36, 0, 27, 0, 27, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 47, 0, 0, 0, 0, 0, 0, 0, 0, 72, -25, 0, 0, -25, 0, 47, 0, 0, 87, -25, 0, 0, -31, 0, 47, 0, 0, 87, -47, 0, 0, -47, 0, 47, 0, 18, 87, -79, 0, 0, -57, 0, 27, 0, 0, 87, -79, 0, -11, -67, 0, 0, 0, 0, 72, -67, 0, 0, -57, 0, 0, 0, 0, 47, -25, 0, 0, -47, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, -39, 47, 18, 0, 0, 0, 0, 0, -31, -57, 47, 18, 0, 0, 0, 0, 0, 0, -67, 36, 0, 0, 0, 0, 0, 0, 0, -57, 0, 0, 0, 0, 0, 0, 0, 0, -57, 0, 0, 0, 0, 0, 0, 0, 0, -47, 0, 0, 0, 0, 0, 0, 0, 0, -25, 36, 0, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 
};

const PROGMEM int8_t Bias1[8][1] = {
	53, 
	-1, 
	-1, 
	46, 
	65, 
	0, 
	61, 
	32, 
};

const PROGMEM int8_t W2[5][8] = {
	0, 0, 0, 0, 0, 39, 0, 0, 
	0, 0, 0, -37, 0, -75, 0, 0, 
	0, 0, 0, 0, 0, -60, 0, -60, 
	0, 0, 0, -75, -37, -37, 0, 0, 
	0, 0, 0, 39, 0, 0, -75, -37, 
};

const PROGMEM int8_t Bias2[5][1] = {
	-74, 
	-10, 
	-9, 
	45, 
	22, 
};

}
