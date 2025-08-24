#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "KeySearchTypes.h"
#include "CudaKeySearchDevice.h"
#include "ptx.cuh"
#include "secp256k1.cuh"

#include "sha256.cuh"
#include "ripemd160.cuh"

#include "secp256k1.h"

#include "CudaHashLookup.cuh"
#include "CudaAtomicList.cuh"
#include "CudaDeviceKeys.cuh"
#include "xor.cuh"

__constant__ unsigned int _INC_X[8];

__constant__ unsigned int _INC_Y[8];

__constant__ unsigned int *_CHAIN[1];

static unsigned int *_chainBufferPtr = NULL;


__device__ void doRMD160FinalRound(const unsigned int hIn[5], unsigned int hOut[5])
{
    const unsigned int iv[5] = {
        0x67452301,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0
    };

    for(int i = 0; i < 5; i++) {
        hOut[i] = endian(hIn[i] + iv[(i + 1) % 5]);
    }
}


/**
 * Allocates device memory for storing the multiplication chain used in
 the batch inversion operation
 */
cudaError_t allocateChainBuf(unsigned int count)
{
    cudaError_t err = cudaMalloc(&_chainBufferPtr, count * sizeof(unsigned int) * 8);

    if(err) {
        return err;
    }

    err = cudaMemcpyToSymbol(_CHAIN, &_chainBufferPtr, sizeof(unsigned int *));
    if(err) {
        cudaFree(_chainBufferPtr);
    }

    return err;
}

void cleanupChainBuf()
{
    if(_chainBufferPtr != NULL) {
        cudaFree(_chainBufferPtr);
        _chainBufferPtr = NULL;
    }
}

/**
 *Sets the EC point which all points will be incremented by
 */
cudaError_t setIncrementorPoint(const secp256k1::uint256 &x, const secp256k1::uint256 &y)
{
    unsigned int xWords[8];
    unsigned int yWords[8];

    x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
    y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

    cudaError_t err = cudaMemcpyToSymbol(_INC_X, xWords, sizeof(unsigned int) * 8);
    if(err) {
        return err;
    }

    return cudaMemcpyToSymbol(_INC_Y, yWords, sizeof(unsigned int) * 8);
}



__device__ void hashPublicKey(const unsigned int *x, const unsigned int *y, unsigned int *digestOut)
{
    unsigned int hash[8];

    sha256PublicKey(x, y, hash);

    // Swap to little-endian
    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);
}

__device__ void hashPublicKeyCompressed(const unsigned int *x, unsigned int yParity, unsigned int *digestOut)
{
    unsigned int hash[8];

    sha256PublicKeyCompressed(x, yParity, hash);

    // Swap to little-endian
    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);
}


__device__ void setResultFound(int idx, bool compressed, unsigned int x[8], unsigned int y[8], unsigned int digest[5])
{
    CudaDeviceResult r;

    r.block = blockIdx.x;
    r.thread = threadIdx.x;
    r.idx = idx;
    r.compressed = compressed;

    for(int i = 0; i < 8; i++) {
        r.x[i] = x[i];
        r.y[i] = y[i];
    }

    doRMD160FinalRound(digest, r.digest);

    atomicListAdd(&r, sizeof(r));
}

__device__ void doIteration(int pointsPerThread, int compression)
{
    unsigned int *chain = _CHAIN[0];
    unsigned int *xPtr = ec::getXPtr();
    unsigned int *yPtr = ec::getYPtr();

    // Multiply together all (_Gx - x) and then invert
    unsigned int inverse[8] = {0,0,0,0,0,0,0,1};
    for(int i = 0; i < pointsPerThread; i++) {
        unsigned int x[8];

        unsigned int digest[5];

        readInt(xPtr, i, x);

        if(compression == PointCompressionType::UNCOMPRESSED || compression == PointCompressionType::BOTH) {
            unsigned int y[8];
            readInt(yPtr, i, y);

            hashPublicKey(x, y, digest);

            if(checkHash(digest)) {
                setResultFound(i, false, x, y, digest);
            }
        }

        if(compression == PointCompressionType::COMPRESSED || compression == PointCompressionType::BOTH) {
            hashPublicKeyCompressed(x, readIntLSW(yPtr, i), digest);

            if(checkHash(digest)) {
                unsigned int y[8];
                readInt(yPtr, i, y);
                setResultFound(i, true, x, y, digest);
            }
        }

        beginBatchAdd(_INC_X, x, chain, i, i, inverse);
    }

    doBatchInverse(inverse);

    for(int i = pointsPerThread - 1; i >= 0; i--) {

        unsigned int newX[8];
        unsigned int newY[8];

        completeBatchAdd(_INC_X, _INC_Y, xPtr, yPtr, i, i, chain, inverse, newX, newY);

        writeInt(xPtr, i, newX);
        writeInt(yPtr, i, newY);
    }
}

__device__ void doIterationWithDouble(int pointsPerThread, int compression)
{
    unsigned int *chain = _CHAIN[0];
    unsigned int *xPtr = ec::getXPtr();
    unsigned int *yPtr = ec::getYPtr();

    // Multiply together all (_Gx - x) and then invert
    unsigned int inverse[8] = {0,0,0,0,0,0,0,1};
    for(int i = 0; i < pointsPerThread; i++) {
        unsigned int x[8];

        unsigned int digest[5];

        readInt(xPtr, i, x);

        // uncompressed
        if(compression == PointCompressionType::UNCOMPRESSED || compression == PointCompressionType::BOTH) {
            unsigned int y[8];
            readInt(yPtr, i, y);
            hashPublicKey(x, y, digest);

            if(checkHash(digest)) {
                setResultFound(i, false, x, y, digest);
            }
        }

        // compressed
        if(compression == PointCompressionType::COMPRESSED || compression == PointCompressionType::BOTH) {

            hashPublicKeyCompressed(x, readIntLSW(yPtr, i), digest);

            if(checkHash(digest)) {

                unsigned int y[8];
                readInt(yPtr, i, y);

                setResultFound(i, true, x, y, digest);
            }
        }

        beginBatchAddWithDouble(_INC_X, _INC_Y, xPtr, chain, i, i, inverse);
    }

    doBatchInverse(inverse);

    for(int i = pointsPerThread - 1; i >= 0; i--) {

        unsigned int newX[8];
        unsigned int newY[8];

        completeBatchAddWithDouble(_INC_X, _INC_Y, xPtr, yPtr, i, i, chain, inverse, newX, newY);

        writeInt(xPtr, i, newX);
        writeInt(yPtr, i, newY);
    }
}

/**
* Performs a single iteration
*/
__global__ void keyFinderKernel(int points, int compression)
{
    doIteration(points, compression);
}

__global__ void keyFinderKernelWithDouble(int points, int compression)
{
    doIterationWithDouble(points, compression);
}

__device__ void privateKeyToPublicKey(const unsigned int *privateKey, unsigned int *publicKeyX, unsigned int *publicKeyY, const unsigned int *basePointsX, const unsigned int *basePointsY)
{
    // Initialize with infinity
    for(int i = 0; i < 8; i++) {
        publicKeyX[i] = 0xFFFFFFFF;
        publicKeyY[i] = 0xFFFFFFFF;
    }

    for (int i = 0; i < 256; i++) {
        unsigned int bit = privateKey[7 - i / 32] & (1 << (i % 32));

        if (bit != 0) {
            unsigned int gx[8];
            unsigned int gy[8];

            for(int j = 0; j < 8; j++) {
                gx[j] = basePointsX[i * 8 + j];
                gy[j] = basePointsY[i * 8 + j];
            }

            if (isInfinity(publicKeyX)) {
                copyBigInt(gx, publicKeyX);
                copyBigInt(gy, publicKeyY);
            } else {
                unsigned int s[8];
                unsigned int dx[8];
                subModP(gx, publicKeyX, dx);
                invModP(dx);
                unsigned int dy[8];
                subModP(gy, publicKeyY, dy);
                mulModP(dx, dy, s);

                unsigned int newX[8];
                unsigned int s2[8];
                squareModP(s, s2);
                subModP(s2, publicKeyX, newX);
                subModP(newX, gx, newX);

                unsigned int newY[8];
                subModP(publicKeyX, newX, dy);
                mulModP(s, dy, newY);
                subModP(newY, publicKeyY, newY);

                copyBigInt(newX, publicKeyX);
                copyBigInt(newY, publicKeyY);
            }
        }
    }
}

__device__ void setResultExport(unsigned int *privateKey, unsigned int *x)
{
    CudaExportResult r;

    for(int i = 0; i < 8; i++) {
        r.privateKey[i] = privateKey[i];
        r.x[i] = x[i];
    }

    atomicListAdd(&r, sizeof(r));
}


__global__ void exportKernel(int mode, unsigned int *startKey, unsigned int *endKey, unsigned int *seed, const unsigned int *basePointsX, const unsigned int *basePointsY)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // mode 0: SEQUENTIAL
    if(mode == 0) {
        unsigned int privateKey[8];

        add_cc(privateKey[7], startKey[7], threadId);
        addc_cc(privateKey[6], startKey[6], 0);
        addc_cc(privateKey[5], startKey[5], 0);
        addc_cc(privateKey[4], startKey[4], 0);
        addc_cc(privateKey[3], startKey[3], 0);
        addc_cc(privateKey[2], startKey[2], 0);
        addc_cc(privateKey[1], startKey[1], 0);
        addc(privateKey[0], startKey[0], 0);

        unsigned int x[8], y[8];
        privateKeyToPublicKey(privateKey, x, y, basePointsX, basePointsY);

        setResultExport(privateKey, x);
    } else {
        struct xor_state s;
        s.x = seed[0] + threadId;
        s.y = seed[1];
        s.z = seed[2];
        s.w = seed[3];

        unsigned int privateKey[8];

        // mode 1: RANDOM_FULL
        if(mode == 1) {
            for(int j = 0; j < 8; j++) {
                privateKey[j] = xor_rand(&s);
            }
        // mode 2: RANDOM_RANGE
        } else {
            unsigned int range[8];
            sub(endKey, startKey, range);

            unsigned int random[8];
            do {
                for(int j = 0; j < 8; j++) {
                    random[j] = xor_rand(&s);
                }
            } while(cmp(random, range) > 0);

            add(startKey, random, privateKey);
        }

        unsigned int x[8], y[8];
        privateKeyToPublicKey(privateKey, x, y, basePointsX, basePointsY);

        setResultExport(privateKey, x);
    }
}