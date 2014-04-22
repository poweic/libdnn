#include <cnn-utility.h>
#include <cuda_profiler_api.h>

/*! Convert each row to a 2D image
 * \param data Each col in data is a feature vector. # of cols = # of data
						     # of rows = image size
 * \param s Size of the image. # of rows in data = s.m x s.n
 */
vector<mat> reshapeVectors2Images(const mat& data, const SIZE s) {

  int nData = data.getCols();
  vector<mat> images(nData);

  for (size_t i=0; i<nData; ++i) {
    images[i].resize(s.m, s.n);

    CCE(cudaMemcpy(images[i].getData(), data.getData() + i * data.getRows(),
	  sizeof(float) * s.m * s.n, cudaMemcpyDeviceToDevice));
  }

  return images;
}

mat reshapeImages2Vectors(const vector<mat>& images) {
  assert(images.size() > 0);

  SIZE s(images[0].getRows(), images[0].getCols());
  mat t_data(s.m * s.n, images.size());

  for (size_t i=0; i<images.size(); ++i)
    CCE(cudaMemcpy(t_data.getData() + i * t_data.getRows(), images[i].getData(),
	  sizeof(float) * images[i].size(), cudaMemcpyDeviceToDevice));

  return t_data;
}


SIZE parseInputDimension(const string &m_by_n) {
  size_t pos = m_by_n.find("x");
  return SIZE(str2int(m_by_n.substr(0, pos)), str2int(m_by_n.substr(pos+1)));
}

__global__ void convn_valid_kernel_with_shm(float *output, float *data, float *kernel, int H, int W, int kH, int kW) { 
  int tx = threadIdx.x;	  /* tx = 0 ~ 31 */
  int ty = threadIdx.y;	  /* ty = 0 ~ 31 */

  int x0 = blockIdx.x*blockDim.x;
  int y0 = blockIdx.y*blockDim.y;
  // Matrix index
  int x = x0 + tx;	  /* x = x ~ x + 31 */
  int y = y0 + ty;	  /* y = y ~ y + 31 */

  extern __shared__ float K[];
  float* D = K + kW * kH;

  // Copy kernel in global memory to shared memory
  if (tx < kW && ty < kH)
    K[(kW - 1 - tx) * kH + (kH - 1 - ty)] = kernel[tx * kH + ty];

  // Copy data in global memory to shared memory
  int nThreads = blockDim.x * blockDim.y;
  int w_step = blockDim.x + kW - 1,
      h_step = blockDim.y + kH - 1;

  int nTotal	=  w_step * h_step;
  int avgToLoad = nTotal / nThreads + 1;

  int tid = tx * blockDim.y + ty;

  // Move data to (x0, y0)
  data += x0 * H + y0;

  for (int i=0; i<avgToLoad; ++i) {
    int id = tid + i * nThreads;

    if (id >= nTotal) break;

    int xx = id / h_step,
	yy = id - xx * h_step; /* OR yy = id % h_step */

    D[id] = data[ xx * H + yy ];
  }
  __syncthreads();

  float sum = 0;
  for (int i = 0; i < kW; ++i)
    for(int j = 0; j < kH; ++j)
      sum += K[ i * kH + j ] * D[ (tx + i) * (blockDim.y + kH - 1) + (ty + j) ]; 

  // vH, vW stands for valid H and valid W
  const int vH = H - kH + 1, vW = W - kW + 1;
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
  else if (type == "full")
    return SIZE(H + kH - 1, W + kW - 1);
  else
    throw std::runtime_error("No such type of convolution");
}

void benchmark() {

  size_t M[] = {32, 64, 128, 256, 384, 512};
  size_t N[] = {3, 6, 9, 12};

  perf::Timer timer;
  const size_t N_TIMES = 200;

  printf("          |");
  for (int j=0; j<sizeof(N) / sizeof(size_t) ; ++j)
    printf("         %3lu x %-3lu          |", N[j], N[j]);
  printf("\n----------+");
  for (int j=0; j<sizeof(N) / sizeof(size_t); ++j)
    printf("----------------------------+");
  printf("\n");

  for (int i=0; i< sizeof(M) / sizeof(size_t) ; ++i) {
    mat x = randn(M[i], M[i]);

    printf("%3lu x %-3lu | ", M[i], M[i]);
    for (int j=0; j<sizeof(N) / sizeof(size_t); ++j) {
      mat kernel = randn(N[j], N[j]);

      timer.start();
      for (int k = 1; k < N_TIMES; ++k) {
	mat z1 = convn(x, kernel, "valid");
      }
      float t1 = timer.getTime();
      timer.reset();

      timer.start();
      for (int k = 1; k < N_TIMES; ++k) {
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

mat convn(const mat& data, const mat& kernel, string type, int N_STREAM) {

  static vector<cudaStream_t> streams(N_STREAM);
  static bool first = true;
  static int counter = 0;

  if (first) {
    first = false;
    for (size_t i=0; i<streams.size(); ++i)
      cudaStreamCreate ( &streams[i] );
  }

  cudaStream_t& stream = streams[counter];
  counter = (counter + 1) % N_STREAM;
  // mat::setCudaStream(stream);

  int H = data.getRows(),
      W = data.getCols(),
      kH = kernel.getRows(),
      kW = kernel.getCols();

  SIZE s = get_convn_size(data, kernel, type);

  mat output(s.m, s.n);

  ALLOCATE_GRIDS_AND_THREADS(output.getRows(), output.getCols());

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
    /* For a data of size 48 x 48 and kernel of size 8 x 8, using shared memory
       can speed up to 4x */

    size_t SHM_SIZE = ( kW * kH + (threads.x + kW - 1) * (threads.y + kH - 1) ) * sizeof(float);
    if (SHM_SIZE > 16 * 1024)
      clog << "\33[35m[Warning]\33[0m Potential excess of Maximum shared memory" << endl;
    // printf("Shared memory size = %lu Bytes (i.e. %f KBytes)\n", SHM_SIZE, (float) SHM_SIZE / 1024);

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
  
  return output;
}

mat xcorrn(const mat& data, const mat& kernel, string type) {
  // TODO
  return mat();
}

__global__ void downsample_kernel(float *dst, float *src, size_t scale, int H, int W) { 
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Matrix index
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;

  int h = H / scale,
      w = W / scale;

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

__global__ void upsample_kernel(float *dst, float *src, size_t scale, int h, int w) { 
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Matrix index
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;

  int H = h * scale,
      W = w * scale;

  if (x >= W || y >= H)
    return;

  dst[x * H + y] = src[(x / scale) * h + (y / scale)];
}

mat downsample(const mat& x, size_t scale) {
  mat output(x.getRows() / scale, x.getCols() / scale);

  ALLOCATE_GRIDS_AND_THREADS(output.getRows(), output.getCols());

  downsample_kernel<<<grids, threads>>>(
      output.getData(),
      x.getData(),
      scale,
      x.getRows(),
      x.getCols());

  CCE(cudaDeviceSynchronize());

  return output;
}

mat upsample(const mat& x, size_t scale) {
  mat output(x.getRows() * scale, x.getCols() * scale);

  ALLOCATE_GRIDS_AND_THREADS(output.getRows(), output.getCols());

  upsample_kernel<<<grids, threads>>>(
      output.getData(),
      x.getData(),
      scale,
      x.getRows(),
      x.getCols());

  CCE(cudaDeviceSynchronize());

  return output;
}

mat rot180(const mat& x) {
  // TODO ROTATE 180 degree (OR create another __global__ called cross_correlation
  return x;
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

void test_convn(string type, int N) {

// #undef matlog
// #define matlog(x) { printf(#x" = [\n"); x.print(); printf("];\n"); }

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

void go_test() {
  mat x = randn(48, 48),
      k = randn(8, 8);

  x *= 100;
  k *= 100;

  perf::Timer timer;
  timer.start();

  mat z_gold = convn(x, k, "valid");

  timer.elapsed();
  timer.reset();
  timer.start();

  mat z = convn(x, k, "valid_shm");

  timer.elapsed();

  float L2norm = nrm2(z - z_gold) / nrm2(z_gold);
  printf("\nL2norm = %.7e ... %s\n", L2norm,
      (L2norm < 1e-6) ? "\33[32m[Passed]\33[0m" : "\33[31m[Failed]\33[0m");
}

// Unit-testing codes for reshapeImages2Vectors & reshapeVectors2Images
void test_reshape() {
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

