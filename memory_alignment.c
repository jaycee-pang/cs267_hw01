#include <immintrin.h>
#include <stdlib.h>

const char* dgemm_desc = "Simple blocked dgemm.";
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 8 
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

// void matmul_vectorized(int lda, double* A, double* B, double* C) {

    // align to cache boundary 


// }





/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        // For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}

    





    // for (int i = 0; i < lda; i++) {
    //     for (int j = 0; j < lda; j++) {

    //         __m256d sum = _mm256_setzero_pd();
    //         for (int k = 0; k < lda; k += 4) {
    //             __m256d a_vec = _mm256_load_pd(&A[i * lda + k]);
    //             __m256d b_vec = _mm256_load_pd(&B[k * lda + j]);
    //             sum = _mm256_fmadd_pd(a_vec, b_vec, sum);
    //         }

    //         double result[4];

    //         _mm256_storeu_pd(result, sum);
    //         for (int k = 0; k < 4; k++) {
    //             C[i * lda + j] += result[k];
    //         }
    //     }
    // }

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    // cache boundary 
    double* A_bound = _mm_malloc(lda * lda * sizeof(double), 64);
    double* B_bound = _mm_malloc(lda * lda * sizeof(double), 64);

    // copy matrices 
    for (int i = 0; i < lda; i++) {
        for (int j = 0; j < lda; j++) {
            A_bound[i*lda + j] = A[i*lda + j];
            B_bound[i*lda + j] = B[i*lda + j];
        }
    }

    // For each block-row of A
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
                do_block(lda, M, N, K, A_bound + i + k * lda, B_bound + k + j * lda, C + i + j * lda);
            }
        }
    }
    _mm_free(A_bound);
    _mm_free(B_bound);
}
