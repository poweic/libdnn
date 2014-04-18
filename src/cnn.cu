#include <cnn.h>

void plotL2normInSemilogy() {
  const float threshold = 1e-6;
  printf("N = length(L2norm);\n");
  printf("threshold = %f * ones(1, N);\n", threshold);
  printf("semilogy(1:N, L2norm, 1:N, threshold);\n");
  printf("axis([1, N, %e, %e]);\n", threshold / 100, threshold * 100);
  printf("legend('Minimum Acceptable Error', 'L2-norm');\n");
}

mat rand(int m, int n) {
  mat x(m, n);
  ext::rand(x);
  return x;
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

mat convn(const mat& data, const mat& kernel, string type) {

  const size_t N = 32;
  dim3 threads(N, N);
  dim3 grid;
  
  int H = data.getRows(),
      W = data.getCols(),
      kH = kernel.getRows(),
      kW = kernel.getCols();

  mat output;
  
  if (type == "same")
    output.resize(H, W);
  else if (type == "valid") {
    int a = max(H - kH + 1, 0),
	b = max(W - kW + 1, 0);

    if (a == 0 || b == 0)
      return mat();

    output.resize(a, b);
  }
  else if (type == "full")
    output.resize(H + kH - 1, W + kW - 1);
  else
    throw std::runtime_error("No such type of convolution");

  grid.x = (unsigned int) ceil((float) output.getCols() / N);
  grid.y = (unsigned int) ceil((float) output.getRows() / N);


  if (type == "same") {
    convn_same_kernel<<<grid, threads>>>(
	output.getData(),
	data.getData(),
	kernel.getData(),
	H, W, kH, kW);
  }
  else if (type == "valid") {
    convn_valid_kernel<<<grid, threads>>>(
	output.getData(),
	data.getData(),
	kernel.getData(),
	H, W, kH, kW);
  }
  else if (type == "full") {
    convn_full_kernel<<<grid, threads>>>(
	output.getData(),
	data.getData(),
	kernel.getData(),
	H, W, kH, kW);
  }

  CCE(cudaDeviceSynchronize());
  
  return output;
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


mat downsample(const mat& x, size_t scale) {
  mat output(x.getRows() / scale, x.getCols() / scale);

  const size_t N = 32;
  dim3 threads(N, N);
  dim3 grid;
  
  grid.x = (unsigned int) ceil((float) output.getCols() / N);
  grid.y = (unsigned int) ceil((float) output.getRows() / N);

  downsample_kernel<<<grid, threads>>>(
      output.getData(),
      x.getData(),
      scale,
      x.getRows(),
      x.getCols());

  CCE(cudaDeviceSynchronize());

  return output;
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

