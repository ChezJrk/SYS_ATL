#include <stdint.h>
#include <stdio.h>
#include <time.h>

// generated from exo
void rank_k_reduce_6x16(
    void *ctxt, int_fast32_t K, const float *A, const float *B, float *C);
void rank_k_reduce_6x16_scheduled(
    void *ctxt, int_fast32_t K, const float *A, const float *B, float *C);

#define K 2048
static float A[6 * K];
static float B[K * 16];
static float C[6 * 16];

void initialize() {
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < K; j++) {
      A[i * K + j] = 3.2;
    }
  }
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < 16; j++) {
      B[i * 16 + j] = 0.2;
    }
  }
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 16; j++) {
      C[i * 16 + j] = 0.0;
    }
  }
  return;
}

int main() {
  clock_t start, end;
  int msec;

  initialize();

  // Calling original matmul
  start = clock();
  for (int i = 0; i < 1000; i++)
    rank_k_reduce_6x16(NULL, K, A, B, C);
  end = clock();

  msec = (end - start) * 1000 / CLOCKS_PER_SEC;
  printf("Time taken for original matmul: %d seconds %d milliseconds\n",
      msec / 1000, msec % 1000);

  // Calling scheduled matmul
  start = clock();
  for (int i = 0; i < 1000; i++)
    rank_k_reduce_6x16_scheduled(NULL, K, A, B, C);
  end = clock();

  msec = (end - start) * 1000 / CLOCKS_PER_SEC;
  printf("Time taken for scheduled matmul: %d seconds %d milliseconds\n",
      msec / 1000, msec % 1000);

  return (0);
}
