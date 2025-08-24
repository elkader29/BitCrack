#ifndef _BRIDGE_H
#define _BRIDGE_H

#include<cuda.h>
#include<cuda_runtime.h>
#include<string>
#include "cudaUtil.h"
#include "secp256k1.h"


void callKeyFinderKernel(int blocks, int threads, int points, bool useDouble, int compression);

void callExportKernel(int blocks, int threads, int mode, unsigned int *startKey, unsigned int *endKey, unsigned int *seed, const unsigned int *basePointsX, const unsigned int *basePointsY);

void waitForKernel();

cudaError_t setIncrementorPoint(const secp256k1::uint256 &x, const secp256k1::uint256 &y);
cudaError_t allocateChainBuf(unsigned int count);
void cleanupChainBuf();

#endif