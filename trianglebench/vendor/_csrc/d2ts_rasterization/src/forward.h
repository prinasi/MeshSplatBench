#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "param_struct.h"
#include "config.h"

namespace FORWARD
{
	__global__ void preprocessCUDA(
		int W, int H, int P, int D, int M, float gamma, bool rich_info,
		bool use_shs, bool use_vertex_color, bool back_culling, dim3 grid,
		const float *__restrict__ viewmatrix,
		const float *__restrict__ projmatrix,
		const float *__restrict__ campos,
		const float *__restrict__ vertex,
		const float *__restrict__ shs,
		const float *__restrict__ opacity,
		int *__restrict__ radii,
		float3 *__restrict__ s_v1_view,
		float3 *__restrict__ s_v2_view,
		float3 *__restrict__ s_v3_view,
		float3 *__restrict__ s_normal_view,
		float *__restrict__ s_depth,
		float3 *__restrict__ s_rgb,
		bool *__restrict__ s_clamped,
		uint32_t *__restrict__ s_tiles_touched,
		uint2 *__restrict__ s_rect_min,
		uint2 *__restrict__ s_rect_max);

	__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
		renderCUDA(
			int W, int H, int C, float gamma, bool rich_info, bool use_vertex_color, bool back_culling,
			float tan_fovx, float tan_fovy,
			const uint2 *__restrict__ ranges,
			const uint32_t *__restrict__ point_list,
			const float3 *__restrict__ s_v1_view,
			const float3 *__restrict__ s_v2_view,
			const float3 *__restrict__ s_v3_view,
			const float3 *__restrict__ s_normal_view,
			const float *__restrict__ feature,
			const float *__restrict__ opacity,
			const float background_depth,
			const float *__restrict__ background,
			float *__restrict__ final_Ts,
			uint32_t *__restrict__ n_contribs,
			float *__restrict__ out_feature,
			float *__restrict__ out_depth,
			float *__restrict__ out_normal,
			float *__restrict__ out_distort,
			float *__restrict__ contrib_sum,
			float *__restrict__ contrib_max);
	
	__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
		renderCUDAResort(
			int W, int H, int C, float gamma, bool rich_info, bool use_vertex_color, bool back_culling,
			float tan_fovx, float tan_fovy,
			const uint2 *__restrict__ ranges,
			const uint32_t *__restrict__ point_list,
			const float3 *__restrict__ s_v1_view,
			const float3 *__restrict__ s_v2_view,
			const float3 *__restrict__ s_v3_view,
			const float3 *__restrict__ s_normal_view,
			const float *__restrict__ feature,
			const float *__restrict__ opacity,
			const float background_depth,
			const float *__restrict__ background,
			float *__restrict__ final_Ts,
			uint32_t *__restrict__ n_contribs,
			float *__restrict__ out_feature,
			float *__restrict__ out_depth,
			float *__restrict__ out_normal,
			float *__restrict__ out_distort,
			float *__restrict__ contrib_sum,
			float *__restrict__ contrib_max);
}
