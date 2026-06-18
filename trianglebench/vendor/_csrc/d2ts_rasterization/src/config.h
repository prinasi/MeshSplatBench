#pragma once

#define ENABLE_VERTEX_COLOR true
#define MAX_CHANNELS 3
#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE / 32)
#define SORT_WINDOW_SIZE 16
#define G_THRES (1.0f / 255.0f)
#define T_THRES 1e-4f
#define ALPHA_THRES 0.99f

// #define DEBUG
