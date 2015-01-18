#ifndef _CNN_UTILITY_H_
#define _CNN_UTILITY_H_

#include <dnn-utility.h>

std::vector<mat> reshapeVectors2Images(const mat& data, const SIZE s);

SIZE get_convn_size(SIZE data, SIZE kernel, ConvType type = FULL);
SIZE get_convn_size(const mat& data, const mat& kernel, ConvType type = FULL);

mat convn(const mat& data, const mat& kernel, ConvType type);

mat downsample(const mat& x, size_t scale, SIZE s);

mat upsample(const mat& x, SIZE s, SIZE img);

#endif // _CNN_UTILITY_H_
