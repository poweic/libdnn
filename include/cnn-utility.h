#ifndef _CNN_UTILITY_H_
#define _CNN_UTILITY_H_

#include <dnn-utility.h>

std::vector<mat> reshapeVectors2Images(const mat& data, const SIZE s);
mat reshapeImages2Vectors(const std::vector<mat>& images);

void showImage(const mat& x);

SIZE get_convn_size(SIZE data, SIZE kernel, ConvType type = FULL);
SIZE get_convn_size(const mat& data, const mat& kernel, ConvType type = FULL);

mat convn(const mat& data, const mat& kernel, ConvType type);
mat convn(const mat& data, const mat& kernel, SIZE s, ConvType type);

mat cross_convn(const mat& dataIn, const mat& dataOut, SIZE imgIn, SIZE imgOut, ConvType type);

std::vector<mat> de_concat(const mat& concated_features, int n);
mat concat(const std::vector<mat>& smalls);

mat downsample(const mat& x, size_t scale, SIZE s);

mat upsample(const mat& x, SIZE s, SIZE img);

mat rot180(const mat& x);

#endif // _CNN_UTILITY_H_
