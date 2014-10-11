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


void gogo() {

  mat x("sheep.mat");
  mat gk = gaussian_kernel(5, 5);

  showImage(x);
  showImage(gk);

  mat y = convn(x, gk, "valid");

  printf("\n\n\n");
  showImage(y);

  y = (y - 0.5f) * 5;
  y = sigmoid(y);
  showImage(y);

  mat z = downsample(y, 2);
  showImage(z);

  mat a = upsample(z, 2);
  showImage(a);

  mat b = upsample(a, 2);
  showImage(b);
}


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

SIZE get_convn_size(SIZE data, SIZE kernel, string type) {
  if (type == "same")
    return data;
  else if (type == "valid" || type == "valid_shm")
    return max(data - kernel + 1, SIZE(0, 0));
  else if (type == "full")
    return data + kernel - 1;
  else
    throw std::runtime_error("No such type of convolution");
}

SIZE get_convn_size(const mat& data, const mat& kernel, string type) {

  int H = data.getRows(),
      W = data.getCols(),
      kH = kernel.getRows(),
      kW = kernel.getCols();

  if (type == "same")
    return SIZE(H, W);
  else if (type == "valid" || type == "valid_shm")
    return SIZE(max(H - kH + 1, 0), max(W - kW + 1, 0));
  else if (type == "full" || type == "full_shm")
    return SIZE(H + kH - 1, W + kW - 1);
  else
    throw std::runtime_error("No such type of convolution");
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

  if (SHM_SIZE > MAX_SHARED_MEMORY_SIZE)
    throw std::runtime_error(RED_ERROR + "Exceeds maximum shared memory available.");

  return SHM_SIZE;
}

/* \brief compute convolution of a batch of data with a kernel.
 * \param data a batch of data, where the batch-size equals to data.getCols()
 * \param kernel the convolutional kernel (i.e. system's impulse response)
 * \param s size of a datum. That is, s.m * s.n = data.getRows()
 * \param type type of convolution. Either "full", "same", or "valid"
 * */
mat convn(const mat& data, const mat& kernel, SIZE s, string type) {

  int H = s.m,
      W = s.n,
      kH = kernel.getRows(),
      kW = kernel.getCols(),
      vH = H - kH + 1,
      vW = W - kW + 1;

  if ( data.getRows() != H * W )
    throw std::runtime_error(DEBUG_STR(data.getRows()) + DEBUG_STR(H) + DEBUG_STR(W));

  int N = data.getCols();

  mat output(vH * vW, N);

  ALLOCATE_GRIDS_AND_THREADS(vH, vW);
  grids.z = N;

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

  if (SHM_SIZE > MAX_SHARED_MEMORY_SIZE)
    throw std::runtime_error(RED_ERROR + "Exceeds maximum shared memory available.");

  convn_valid_kernel_with_shm<<< grids, threads, SHM_SIZE, 0 >>>(
      output.getData(),
      data.getData(),
      kernel.getData(),
      H, W, kH, kW);

  CCE(cudaPeekAtLastError());
  CCE(cudaDeviceSynchronize());

  return output;
}

mat convn(const mat& data, const mat& kernel, string type) {

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

  if (type == "same") {
    convn_same_kernel<<< grids, threads, 0, stream >>>(
	output.getData(),
	data.getData(),
	kernel.getData(),
	H, W, kH, kW);
  }
  else if (type == "valid") {
    convn_valid_kernel<<< grids, threads, 0, stream >>>(
	output.getData(),
	data.getData(),
	kernel.getData(),
	H, W, kH, kW);
  }
  else if (type == "valid_shm") {

    size_t SHM_SIZE = getSuitableShmConfig(grids, threads, kH, kW);

    convn_valid_kernel_with_shm<<< grids, threads, SHM_SIZE, stream >>>(
	output.getData(),
	data.getData(),
	kernel.getData(),
	H, W, kH, kW);
  }
  else if (type == "full") {
    convn_full_kernel<<< grids, threads, 0, stream >>>(
	output.getData(),
	data.getData(),
	kernel.getData(),
	H, W, kH, kW);
  }
  else if (type == "full_shm") {

    size_t SHM_SIZE = getSuitableShmConfig(grids, threads, kH, kW);

    convn_full_kernel_with_shm<<< grids, threads, SHM_SIZE, stream >>>(
	output.getData(),
	data.getData(),
	kernel.getData(),
	H, W, kH, kW);
  }

  CCE(cudaPeekAtLastError());
  CCE(cudaDeviceSynchronize());
  
  return output;
}


/*mat xcorrn(const mat& data, const mat& kernel, string type) {
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

mat concat(const vector<mat>& smalls) {
  int nFeatures = smalls.size(),
      img_size  = smalls[0].getRows(),
      batchSize = smalls[0].getCols();

  mat big(img_size * nFeatures, batchSize);

  int MAP_SIZE = smalls[0].size();

  for (int i=0; i<nFeatures; ++i)
    memcpy2D(big, smalls[i], 0, 0, img_size, batchSize, i * img_size, 0);

  CCE(cudaDeviceSynchronize());

  return big;
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

mat rot180(const mat& x) {
  // FIXME 我偷懶。我先講求正確性，丟到host用double for loop轉，轉完再塞回device

  int rows = x.getRows(),
      cols = x.getCols();

  hmat h_x(x), h_y(rows, cols);

  for (size_t i=0; i<rows; ++i) {
    for (size_t j=0; j<cols; ++j) {
      h_y(i, j) = h_x(rows - 1 - i, cols - 1 - j);
    }
  }

  return (mat) h_y;
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

/* Codes for unit-testing 
 * 
 * 
 */

void plotL2normInSemilogy() {
  const float threshold = 1e-6;
  printf("N = length(L2norm);\n");
  printf("threshold = %f * ones(1, N);\n", threshold);
  printf("semilogy(1:N, L2norm, 1:N, threshold);\n");
  printf("axis([1, N, %e, %e]);\n", threshold / 100, threshold * 100);
  printf("legend('Minimum Acceptable Error', 'L2-norm');\n");
}

void test_downsample() {

  int counter = 1;

  for (int i = 0; i<20; ++i) {
    int M = rand() % 35 + 69,
	N = rand() % 43 + 28;

    mat x = rand(M, N);

    for (int scale = 2; scale < 10; ++scale) {
      mat y = downsample(x, scale);

      matlog(x);
      matlog(y);

      printf("tmp = convn(x, ones(%d) / (%d ^ 2), 'valid');\n", scale, scale);
      printf("y_gold = tmp(1:%d:end, 1:%d:end);\n", scale, scale);
      printf("delta = y - y_gold;\n");
      printf("L2norm(%d) = norm(delta(:)) / norm(y_gold(:)) / 2;\n", counter++);
    }
  }

  plotL2normInSemilogy();
}

void test_convn(string type) {

// #undef matlog
// #define matlog(x) { printf(#x" = [\n"); x.print(); printf("];\n"); }

  const int N = 10000;

  for (int i=0; i<N; ++i) {
    int W = rand() % 50 + 5,
	H = rand() % 50 + 5,
	kW = rand() % (W-1) + 1,
	kH = rand() % (H-1) + 1;

    mat data = rand(W, H);
    mat kernel = rand(kW, kH);

    mat z = convn(data, kernel, type);
    matlog(data);
    matlog(kernel);
    matlog(z);

    printf("z_gold = convn(data, kernel, '%s');\n", type.c_str());
    printf("delta = z_gold - z;\n");
    printf("L2norm(%d) = norm(delta(:)) / norm(z_gold(:)) / 2;\n", i + 1);
  }

  plotL2normInSemilogy();
}

void test_convn_with_and_without_shm(string type, const int N) {

  bool all_pass = true;

  for (int i=0; i<N; ++i) {
    int W = rand() % 120 + 40,
	H = rand() % 120 + 40,
	kW = rand() % 20 + 4,
	kH = rand() % 20 + 4;

    mat x = randn(H, W),
	k = randn(kH, kW);

    mat z_gold = convn(x, k, type);
    mat z = convn(x, k, type + "_shm");

    float L2norm = nrm2(z - z_gold) / nrm2(z_gold);
    printf("L2norm = %.7e ...", L2norm);

    if (L2norm < 1e-6) {
      printf("\33[32m[Passed]\33[0m\n");
    }
    else {
      printf("\33[31m[Failed]\33[0m\n");
      all_pass = false;
    }
  }

  if (all_pass)
    printf("\33[32m !!! Congrats! ALL %d test cases PASSED !!! \33[0m\n", N);
  else
    printf("\33[31m !!! Oh oh! some test cases FAILED !!! \33[0m\n");
}

void test_valid_shm_vs_valid_2() {
  mat x = randn(200, 200);
  for (int i=5; i<77; ++i)  {
    for (int j=5; j<77; ++j) {
      printf("kernel: %d x %d\t", i, j);
      mat kernel = randn(i, j);

      mat z1 = convn(x, kernel, "valid_shm");
      mat z2 = convn(x, kernel, "valid");

      float a = nrm2(z1 - z2),
	    b = nrm2(z2);
      float l2error = a / b / 2;

      assert(l2error == l2error);
      printf("l2error = %.7e / %.7e / 2 = %.7e \t", a, b, l2error);
      if (l2error > 1e-6)
	printf("\33[31m[FAILED]\33[0m\n");
      else
	printf("\33[32m[PASSED]\33[0m\n");
      printf("\n");
    }
  }
}

// Unit-testing codes for reshapeImages2Vectors & reshapeVectors2Images
void test_reshape_images_between_vectors() {
  int batch_size = 12;
  SIZE img(4, 5);
  mat x = randn(img.m * img.n, batch_size);

  vector<mat> images = reshapeVectors2Images(x, img);

  for (size_t i=0; i<images.size(); ++i)
    matlog(images[i]);

  mat y = reshapeImages2Vectors(images);

  matlog(x);
  matlog(y);

  matlog(x-y);
}


void benchmark_valid_and_valid_shm() {

  size_t M[] = {32, 64, 128, 256, 384, 512};
  size_t N[] = {3, 6, 9, 12};

  perf::Timer timer;
  const size_t N_TIMES = 200;

  printf("          |");
  for (size_t j=0; j<sizeof(N) / sizeof(size_t) ; ++j)
    printf("         %3lu x %-3lu          |", N[j], N[j]);
  printf("\n----------+");
  for (size_t j=0; j<sizeof(N) / sizeof(size_t); ++j)
    printf("----------------------------+");
  printf("\n");

  for (size_t i=0; i< sizeof(M) / sizeof(size_t) ; ++i) {
    mat x = randn(M[i], M[i]);

    printf("%3lu x %-3lu | ", M[i], M[i]);
    for (size_t j=0; j<sizeof(N) / sizeof(size_t); ++j) {
      mat kernel = randn(N[j], N[j]);

      timer.start();
      for (size_t k = 1; k < N_TIMES; ++k) {
	mat z1 = convn(x, kernel, "valid");
      }
      float t1 = timer.getTime();
      timer.reset();

      timer.start();
      for (size_t k = 1; k < N_TIMES; ++k) {
	mat z2 = convn(x, kernel, "valid_shm");
      }
      float t2 = timer.getTime();
      printf("%7.2f , %7.2f \33[34m->\33[0m %4.1fx", t1, t2, t1 / t2);
      timer.reset();

      printf(" | ");
    }
    printf("\n");
  }
}

void benchmark_batch_convn() {

  // Achieve 3.98x speed up for batch size 128, 3.41x speed up for batch size 32
  const size_t N = 100;

  perf::Timer timer1, timer2;

  for (size_t i=0; i<N; ++i) {
    int nImages = 128;
    int m = rand() % 17 + 23;
    int n = rand() % 13 + 27;
    int kh = 5 + rand() % 2; // rand() % 22 + 4;
    int kw = 5 + rand() % 2; // rand() % 22 + 8;

    SIZE s(m, n);
    mat X = randn(m * n, nImages);
    mat kernel = randn(kh, kw);

    // Slow method
    timer1.start();
    vector<mat> images = reshapeVectors2Images(X, s);
    vector<mat> z_golds(nImages);

    for (int i=0; i<nImages; ++i)
      z_golds[i] = convn(images[i], kernel, "valid_shm");

    mat z_gold = reshapeImages2Vectors(z_golds);
    timer1.stop();

    // Fast method
    timer2.start();
    mat z = convn(X, kernel, s, "valid_shm");
    timer2.stop();

    printf("# of images = %3d, images size: %3d x %-3d, kernel: %3d x %-3d\t", 
	nImages, m, n, kh, kw);

    if (z.getRows() == z_gold.getRows() && z.getCols() == z_gold.getCols()) {
      float l2norm = nrm2(z - z_gold) / nrm2(z_gold) / 2;
      printf("L2norm = %.7e\t", l2norm);
      if (l2norm < 1e-6)
	printf("\33[32m[PASSED]\33[0m\n");
      else
	printf("\33[31m[FAILED]\33[0m\n");
    }
    else
	printf("\33[35m[DIMENSION MISMATCH]\33[0m\n");
  }

  float t1 = timer1.getTime(),
	t2 = timer2.getTime();

  printf("%.4f => %.4f. %.4f x speed up !!\n", t1, t2, t1 / t2);
}

