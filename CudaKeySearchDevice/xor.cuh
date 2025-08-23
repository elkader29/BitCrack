#ifndef _XOR_CUH
#define _XOR_CUH

struct xor_state {
    unsigned int x, y, z, w;
};

__device__ unsigned int xor_rand(struct xor_state *state)
{
    unsigned int t = state->x ^ (state->x << 11);
    state->x = state->y;
    state->y = state->z;
    state->z = state->w;
    state->w = (state->w ^ (state->w >> 19)) ^ (t ^ (t >> 8));

    return state->w;
}

#endif
