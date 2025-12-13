#include <cuda_runtime.h>
#include <cstdio>
int main() {
  int n = 0;
  cudaError_t e = cudaGetDeviceCount(&n);
  printf("cudaGetDeviceCount: %s, n=%d\n", cudaGetErrorString(e), n);
  if(n>0){
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("GPU0: %s, cc=%d.%d\n", p.name, p.major, p.minor);
  }
  return 0;
}
