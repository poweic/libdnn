// Copyright 2013-2014 [Author: Po-Wei Chou]
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cnn-utility.h>
#include <cuda_profiler_api.h>
#define DEBUG_STR(x) ("\33[33m"#x"\33[0m = " + to_string(x) + "\t")
#define MAX_SHARED_MEMORY_SIZE (48 * 1024)

#define __CUDA_CONSTANTS__ \
  const int nThreads = blockDim.x * blockDim.y;\
  const int tx = threadIdx.x;\
  const int ty = threadIdx.y;\
  const int tid = tx + blockDim.x * ty;\
  const int x0 = blockIdx.x * blockDim.x;\
  const int y0 = blockIdx.y * blockDim.y;\
  const int x = x0 + tx;\
  const int y = y0 + ty;

/*! Convert each row to a 2D image
 * \param data Each col in data is a feature vector. # of cols = # of data
						     # of rows = image size
 * \param s Size of the image. # of rows in data = s.m x s.n
 */
vector<mat> reshapeVectors2Images(const mat& data, const SIZE s) {

  size_t nData = data.getCols();
  vector<mat> images(nData);

  for (size_t i=0; i<nData; ++i) {
    images[i].resize(s.m, s.n);

    CCE(cudaMemcpy(images[i].getData(), data.getData() + i * data.getRows(),
	  sizeof(float) * s.m * s.n, cudaMemcpyDeviceToDevice));
  }
  CCE(cudaDeviceSynchronize());

  return images;
}

mat reshapeImages2Vectors(const vector<mat>& images) {
  assert(images.size() > 0);

  SIZE s(images[0].getRows(), images[0].getCols());
  mat t_data(s.m * s.n, images.size());

  for (size_t i=0; i<images.size(); ++i) {
    CCE(cudaMemcpy(t_data.getData() + i * t_data.getRows(), images[i].getData(),
	  sizeof(float) * images[i].size(), cudaMemcpyDeviceToDevice));
  }
  CCE(cudaDeviceSynchronize());

  return t_data;
}

string getColorCode(float n) {
  int x = 232 + n * (256-232);
  return "\33[38;5;" + to_string(x) + "m";
}

mat gaussian_kernel(int h, int w) {
  static mat gk("gk.mat");
  if (h != 5 || w != 5)
    throw std::runtime_error("NO SUCH Gaussian Kernel");

  return gk / 273.f;
}

void showImage(const mat& x) {

  int rows = x.getRows(),
      cols = x.getCols();

  hmat h_x(x);

  for (size_t i=0; i<rows; ++i) {
    for (size_t j=0; j<cols; ++j)
      cout << getColorCode(h_x(i, j)) << "◼" << " ";
    cout << endl;
  }
  cout << "\33[0m" << endl;
}

SIZE parseInputDimension(const string &m_by_n) {
  size_t pos = m_by_n.find("x");

  if (pos == string::npos)
    throw std::runtime_error(RED_ERROR + "Please use --input-dim like this: 32x32");

  return SIZE(str2int(m_by_n.substr(0, pos)), str2int(m_by_n.substr(pos+1)));
}

__device__ void load_kernel_into_shm(float* const K, const float* const kernel, int kH, int kW, int tid, int nThreads) {

  // Copy kernel in global memory to shared memory
  int nTotal = kW * kH;
  int avgToLoad = (kW * kH) / nThreads + 1;

  for (int i=0; i<avgToLoad; ++i) {
    int id = tid + i * nThreads;

    if (id >= nTotal) break;

    int xx = id / kH,
	yy = id % kH;

    if (xx >= kW || yy >= kH) continue;

    K[(kW - 1 - xx) * kH + (kH - 1 - yy)] = kernel[xx * kH + yy];
  }
}

__global__ void convn_valid_kernel_with_shm2(float *output, const float *data,
  float *kernel, const int H, const int W, const int kH, const int kW) { 

  __CUDA_CONSTANTS__;

  // vH, vW stands for valid H and valid W
  const int vH = H - kH + 1, vW = W - kW + 1;

  kernel += blockIdx.z * kH * kW;
  output += blockIdx.z * vH * vW;

  extern __shared__ float K[];

  // Copy kernel in global memory to shared memory
  load_kernel_into_shm(K, kernel, kH, kW, tid, nThreads);

  // Copy data in global memory to shared memory
  float* D = K + kW * kH;
  int WIDTH_STEP = blockDim.x + kW - 1,
      HEIGHT_STEP = blockDim.y + kH - 1;

  int nTotal = WIDTH_STEP * HEIGHT_STEP;
  int avgToLoad  = nTotal / nThreads + 1;

  for (int i=0; i<avgToLoad; ++i) {
    int id = tid + i * nThreads;

    if (id >= nTotal) break;

    int xx = id / HEIGHT_STEP + x0,
	yy = id % HEIGHT_STEP + y0;

    if (xx >= W || yy >= H) continue;

    D[id] = data[ xx * H + yy ];
  }
  __syncthreads();

  float sum = 0;
  for (int i = 0; i < kW; ++i)
    for(int j = 0; j < kH; ++j)
      sum += K[ i * kH + j ] * D[ (tx + i) * HEIGHT_STEP + (ty + j) ]; 

  if (x < vW && y < vH)
    output[ x * vH + y ] = sum;
} 

__global__ void convn_valid_kernel_with_shm(float *output, const float *data,
  float *kernel, const int H, const int W, const int kH, const int kW) { 

  __CUDA_CONSTANTS__;

  // vH, vW stands for valid H and valid W
  const int vH = H - kH + 1, vW = W - kW + 1;

  data += blockIdx.z * H * W;
  output += blockIdx.z * vH * vW;

  extern __shared__ float K[];

  // Copy kernel in global memory to shared memory
  load_kernel_into_shm(K, kernel, kH, kW, tid, nThreads);

  // Copy data in global memory to shared memory
  float* D = K + kW * kH;
  int WIDTH_STEP = blockDim.x + kW - 1,
      HEIGHT_STEP = blockDim.y + kH - 1;

  int nTotal = WIDTH_STEP * HEIGHT_STEP;
  int avgToLoad  = nTotal / nThreads + 1;

  for (int i=0; i<avgToLoad; ++i) {
    int id = tid + i * nThreads;

    if (id >= nTotal) break;

    int xx = id / HEIGHT_STEP + x0,
	yy = id % HEIGHT_STEP + y0;

    if (xx >= W || yy >= H) continue;

    D[id] = data[ xx * H + yy ];
  }
  __syncthreads();

  float sum = 0;
  for (int i = 0; i < kW; ++i)
    for(int j = 0; j < kH; ++j)
      sum += K[ i * kH + j ] * D[ (tx + i) * HEIGHT_STEP + (ty + j) ]; 

  if (x < vW && y < vH)
    output[ x * vH + y ] = sum;
} 

__global__ void convn_valid_kernel(float *output, float *data, float *kernel, int H, int W, int kH, int kW) { 
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Matrix index
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;

  // vH, vW stands for valid H and valid W
  const int vH = H - kH + 1,
	    vW = W - kW + 1;

  if (x >= vW || y >= vH)
    return;

  x += kW - 1;
  y += kH - 1;

  float sum = 0; 
  for (int i = 0; i < kW; ++i)
    for(int j = 0; j < kH; ++j)
      sum += kernel[ i * kH + j ] * data[ (x - i) * H + (y - j) ]; 

  x -= kW - 1;
  y -= kH - 1;

  output[ x * vH + y ] = sum;
} 

__global__ void convn_same_kernel(float *output, float *data, float *kernel, int H, int W, int kH, int kW) { 
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Matrix index
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;

  if (x >= W || y >= H)
    return;

  const int i0 = kW / 2, j0 = kH / 2;

  float sum = 0; 
  for (int i = 0; i < kW; ++i) {
    for(int j = 0; j < kH; ++j) {
      int ii = x - i + i0, jj = y - j + j0;

      if ( ii < 0 || ii >= W || jj < 0 || jj >= H )
	continue;

      sum += kernel[ i * kH + j ] * data[ ii * H + jj ]; 
    }
  }

  output[x * H + y] = sum;
} 

__global__ void convn_full_kernel_with_shm(float *output, float *data, float *kernel, int H, int W, int kH, int kW) { 

  __CUDA_CONSTANTS__;

  // fH, fW stands for full H and full W
  const int fH = H + kH - 1, fW = W + kW - 1;

  data += blockIdx.z * H * W;
  output += blockIdx.z * fH * fW;

  extern __shared__ float K[];

  // Copy kernel in global memory to shared memory
  load_kernel_into_shm(K, kernel, kH, kW, tid, nThreads);

  // Copy data in global memory to shared memory
  float* D = K + kW * kH;
  int WIDTH_STEP = blockDim.x + kW - 1,
      HEIGHT_STEP = blockDim.y + kH - 1;

  int nTotal = WIDTH_STEP * HEIGHT_STEP;
  int avgToLoad  = nTotal / nThreads + 1;

  for (int i=0; i<avgToLoad; ++i) {
    int id = tid + i * nThreads;

    if (id >= nTotal) break;

    int xx = id / HEIGHT_STEP + x0 - (kW - 1),
	yy = id % HEIGHT_STEP + y0 - (kH - 1);

    if (xx < 0 || xx >= W || yy < 0 || yy >= H)
      D[id] = 0;
    else 
      D[id] = data[ xx * H + yy ];
  }
  __syncthreads();
  
  if (x >= fW || y >= fH)
    return;

  float sum = 0; 
  for (int i = 0; i < kW; ++i)
    for(int j = 0; j < kH; ++j)
      sum += K[ i * kH + j ] * D[ (tx + i) * HEIGHT_STEP + (ty + j) ]; 

  output[ x * fH + y ] = sum;
}

__global__ void convn_full_kernel(float *output, float *data, float *kernel, int H, int W, int kH, int kW) { 
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Matrix index
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;

  // fH, fW stands for full H and full W
  const int fH = H + kH - 1,
	    fW = W + kW - 1;

  if (x >= fW || y >= fH)
    return;

  float sum = 0; 
  for (int i = 0; i < kW; ++i) {
    for(int j = 0; j < kH; ++j) {
      int ii = x - i, jj = y - j;

      if ( ii < 0 || ii >= W || jj < 0 || jj >= H )
	continue;

      sum += kernel[ i * kH + j ] * data[ ii * H + jj ]; 
    }
  }

  output[ x * fH + y ] = sum;
}

SIZE get_convn_size(SIZE data, SIZE kernel, ConvType type) {
  switch (type) {
    case SAME:
    case SAME_SHM:
      return data;
    case VALID:
    case VALID_SHM:
      return max(data - kernel + 1, SIZE(0, 0));
    case FULL:
    case FULL_SHM:
      return data + kernel - 1;
    default:
      throw std::runtime_error("Unknown type of convolution.");
  };
}

SIZE get_convn_size(const mat& data, const mat& kernel, ConvType type) {

  SIZE dSize(data.getRows(), data.getCols());
  SIZE kSize(kernel.getRows(), kernel.getCols());

  return get_convn_size(dSize, kSize, type);
}

size_t getSuitableShmConfig(dim3 &grids, dim3 &threads, int kH, int kW) {
  size_t SHM_SIZE = ( kW * kH + (threads.x + kW - 1) * (threads.y + kH - 1) ) * sizeof(float);

  while ( SHM_SIZE > MAX_SHARED_MEMORY_SIZE && threads.x * threads.y >= 32 ) {
    if ( threads.x >= threads.y ) {
      threads.x /= 2;
      grids.x *= 2;
    }
    else {
      threads.y /= 2;
      grids.y *= 2;
    }

    SHM_SIZE = ( kW * kH + (threads.x + kW - 1) * (threads.y + kH - 1) ) * sizeof(float);
  }
  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

  if (SHM_SIZE > MAX_SHARED_MEMORY_SIZE) {
    char buf[512];
    sprintf(buf, "Exceeds maximum shared memory available. (%d bytes)\n"
	"kernel = (%d, %d), grids = (%u, %u, %u), threads = (%u, %u, %u) "
	" => %lu bytes of shared memory needed.", MAX_SHARED_MEMORY_SIZE, kH, kW,
	grids.x, grids.y, grids.z, threads.x, threads.y, threads.z, SHM_SIZE);
    throw std::runtime_error(RED_ERROR + to_string(buf));
  }

  return SHM_SIZE;
}

// Single data with Multiple kernels
mat convn_2(const mat& data, const mat& kernels, SIZE k) {

  SIZE d(data.getRows(), data.getCols());

  SIZE imgOut = get_convn_size(d, k, VALID_SHM);

  if ( kernels.getRows() != k.m * k.n )
    throw std::runtime_error(RED_ERROR + DEBUG_STR(data.getRows()) + DEBUG_STR(k.m) + DEBUG_STR(k.n));

  // i.e. batch_size
  int N = kernels.getCols();

  mat output(imgOut.m * imgOut.n, N);

  ALLOCATE_GRIDS_AND_THREADS(imgOut.m, imgOut.n);
  grids.z = N;

  size_t SHM_SIZE = getSuitableShmConfig(grids, threads, k.m, k.n);

  convn_valid_kernel_with_shm2<<< grids, threads, SHM_SIZE, 0 >>>(
      output.getData(),
      data.getData(),
      kernels.getData(),
      d.m, d.n, k.m, k.n);

  CCE(cudaPeekAtLastError());

  return output;
}

/* \brief compute convolution of a batch of data with a kernel.
 * \param data a batch of data, where the batch-size equals to data.getCols()
 * \param kernel the convolutional kernel (i.e. system's impulse response)
 * \param imgIn size of a datum. That is, imgIn.m * imgIn.n = data.getRows()
 * \param type type of convolution. Either "full", "same", or "valid"
 * */
mat convn(const mat& data, const mat& kernel, SIZE imgIn, ConvType type) {

  int H = imgIn.m,
      W = imgIn.n,
      kH = kernel.getRows(),
      kW = kernel.getCols();

  SIZE imgOut = get_convn_size(imgIn, SIZE(kH, kW), type);

  if ( data.getRows() != H * W )
    throw std::runtime_error(RED_ERROR + DEBUG_STR(data.getRows()) + DEBUG_STR(H) + DEBUG_STR(W));

  // i.e. batch_size
  int N = data.getCols();

  mat output(imgOut.m * imgOut.n, N);

  ALLOCATE_GRIDS_AND_THREADS(imgOut.m, imgOut.n);
  grids.z = N;

  size_t SHM_SIZE = getSuitableShmConfig(grids, threads, kH, kW);

  switch (type) {
    case SAME:
      // TODO
      break;
    case SAME_SHM:
      // TODO
      break;
    case VALID:
      // TODO
      break;
    case VALID_SHM:
      convn_valid_kernel_with_shm<<< grids, threads, SHM_SIZE, 0 >>>(
	  output.getData(),
	  data.getData(),
	  kernel.getData(),
	  H, W, kH, kW);
      break;
    case FULL:
      // TODO
      break;
    case FULL_SHM:
      convn_full_kernel_with_shm<<< grids, threads, SHM_SIZE, 0 >>>(
	  output.getData(),
	  data.getData(),
	  kernel.getData(),
	  H, W, kH, kW);
      break;
  }

  CCE(cudaPeekAtLastError());
  // CCE(cudaDeviceSynchronize());

  return output;
}

mat convn(const mat& data, const mat& kernel, ConvType type) {

  const size_t N_STREAM = 4;
  static vector<cudaStream_t> streams(N_STREAM);
  static bool first = true;
  static int counter = 0;

  if (first) {
    first = false;
    for (size_t i=0; i<streams.size(); ++i)
      cudaStreamCreate ( &streams[i] );
  }

  int H = data.getRows(),
      W = data.getCols(),
      kH = kernel.getRows(),
      kW = kernel.getCols();

  SIZE s = get_convn_size(data, kernel, type);

  mat output(s.m, s.n);
  ALLOCATE_GRIDS_AND_THREADS(output.getRows(), output.getCols());

  // printf("stream-id #%lu, grid: (%lu, %lu), threads: (%lu, %lu)\n", counter, grids.x, grids.y, threads.x, threads.y);

  // cudaStream_t& stream = streams[counter];
  cudaStream_t stream = 0;
  counter = (counter + 1) % N_STREAM;

  switch (type) {
    case SAME:
      convn_same_kernel<<< grids, threads, 0, stream >>>(
	  output.getData(),
	  data.getData(),
	  kernel.getData(),
	  H, W, kH, kW);
      break;
    case SAME_SHM:
      // TODO
      break;
    case VALID:
      convn_valid_kernel<<< grids, threads, 0, stream >>>(
	  output.getData(),
	  data.getData(),
	  kernel.getData(),
	  H, W, kH, kW);
      break;
    case VALID_SHM: {
      size_t SHM_SIZE = getSuitableShmConfig(grids, threads, kH, kW);

      convn_valid_kernel_with_shm<<< grids, threads, SHM_SIZE, stream >>>(
	  output.getData(),
	  data.getData(),
	  kernel.getData(),
	  H, W, kH, kW);
    } break;
    case FULL:
      convn_full_kernel<<< grids, threads, 0, stream >>>(
	  output.getData(),
	  data.getData(),
	  kernel.getData(),
	  H, W, kH, kW);
      break;
      
    case FULL_SHM: {
    size_t SHM_SIZE = getSuitableShmConfig(grids, threads, kH, kW);

    convn_full_kernel_with_shm<<< grids, threads, SHM_SIZE, stream >>>(
	output.getData(),
	data.getData(),
	kernel.getData(),
	H, W, kH, kW);
    } break;
    default:
      throw std::runtime_error(RED_ERROR + "Unknown convolution type");
  }

  CCE(cudaPeekAtLastError());
  CCE(cudaDeviceSynchronize());
  
  return output;
}

/*mat xcorrn(const mat& data, const mat& kernel, ConvType type) {
  // TODO
  return mat();
}*/

// Perform the reverse of concat
vector<mat> de_concat(const mat& big, int N) {

  int batch_size = big.getCols();
  vector<mat> smalls(N);

  int MAP_SIZE = big.size() / N;

  SIZE s(MAP_SIZE / batch_size, batch_size);
  
  for (int i=0; i<N; ++i) {
    smalls[i].resize(s.m, s.n);
    memcpy2D(smalls[i], big, i * s.m, 0, s.m, s.n, 0, 0);
  }

  CCE(cudaDeviceSynchronize());

  return smalls;
}

__global__ void downsample_kernel(float *dst, float *src, size_t scale, int H, int W) { 
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Matrix index
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;

  int h = H / scale,
      w = W / scale;

  src += blockIdx.z * H * W;
  dst += blockIdx.z * h * w;

  if (x >= w || y >= h)
    return;

  float sum;
  for (int i=0; i<scale; ++i) {
    for (int j=0; j<scale; ++j) {
      if ( x*scale + i < W && y*scale + j < H )
	sum += src[(x*scale + i) * H + (y*scale + j)];
    }
  }

  dst[x * h + y] = sum / (scale * scale);
}

__global__ void upsample_kernel(float *dst, float *src, int h, int w, int H, int W) { 
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Matrix index
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;

  src += blockIdx.z * h * w;
  dst += blockIdx.z * H * W;

  int scale = H / h;

  if (x >= W || y >= H)
    return;

  int sx = x / scale, sy = y / scale;
  if (sx == w) --sx;
  if (sy == h) --sy;

  dst[x * H + y] = src[sx * h + sy];
}

mat downsample(const mat& x, size_t scale, SIZE s) {
  int batch_size = x.getCols();

  int H = s.m,
      W = s.n,
      h = H / scale,
      w = W / scale;

  if ( x.getRows() != H * W )
    throw std::runtime_error(DEBUG_STR(x.getRows()) + DEBUG_STR(H) + DEBUG_STR(W));

  mat output(h * w, batch_size);

  ALLOCATE_GRIDS_AND_THREADS(h, w);
  grids.z = batch_size;

  downsample_kernel<<<grids, threads>>>(
      output.getData(),
      x.getData(),
      scale, H, W);

  CCE(cudaDeviceSynchronize());

  return output;
}

mat downsample(const mat& x, size_t scale) {
  int H = x.getRows(),
      W = x.getCols(),
      h = H / scale,
      w = W / scale;

  mat output(h, w);

  ALLOCATE_GRIDS_AND_THREADS(h, w);

  downsample_kernel<<<grids, threads>>>(
      output.getData(),
      x.getData(),
      scale, H, W);

  CCE(cudaDeviceSynchronize());

  return output;
}

mat upsample(const mat& x, size_t scale) {
  return upsample(x, SIZE(x.getRows() * scale, x.getCols() * scale));
}

mat upsample(const mat& x, SIZE s, SIZE img) {

  int batch_size = x.getCols();
  int H = s.m, 
      W = s.n,
      h = img.m,
      w = img.n;

  if ( x.getRows() != img.m * img.n )
    throw std::runtime_error(DEBUG_STR(x.getRows()) + DEBUG_STR(img.m) + DEBUG_STR(img.n));

  mat output(H * W, batch_size);
  ALLOCATE_GRIDS_AND_THREADS(H, W);
  grids.z = batch_size;

  upsample_kernel<<<grids, threads>>>(
      output.getData(),
      x.getData(),
      h, w, H, W);

  CCE(cudaDeviceSynchronize());

  return output;
}

mat upsample(const mat& x, SIZE s) {
  mat output(s.m, s.n);

  ALLOCATE_GRIDS_AND_THREADS(output.getRows(), output.getCols());

  upsample_kernel<<<grids, threads>>>(
      output.getData(),
      x.getData(),
      x.getRows(),
      x.getCols(),
      output.getRows(),
      output.getCols());

  CCE(cudaDeviceSynchronize());

  return output;
}

template <typename T>
__global__ void rot180_kernel(T *odata, const T *idata, const int rows, const int cols) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < cols && y < rows)
    odata[x*rows + y] = idata[(cols - 1 - x) * rows+ (rows - 1 - y)];
}

mat rot180(const mat& x) {

  int rows = x.getRows(),
      cols = x.getCols();

  mat y(rows, cols);
  ALLOCATE_GRIDS_AND_THREADS(rows, cols);
  rot180_kernel<<<grids, threads>>>(y.getData(), x.getData(), rows, cols);

  return y;
}

/* ! \brief Sum all the elements in a matrix.
 * \fn sum_all(const device_matrix<float>& x)
 * \param x matrix x to be sum
 * return the result in host memory.
 */
float sum_all(const mat& x) {
  int r = x.getRows(),
      c = x.getCols();
  mat d_s = mat(1, r, 1) * x * mat(c, 1, 1);

  float s;
  CCE(cudaMemcpy(&s, d_s.getData(), sizeof(float), cudaMemcpyDeviceToHost));
  CCE(cudaDeviceSynchronize());
  return s;
}
