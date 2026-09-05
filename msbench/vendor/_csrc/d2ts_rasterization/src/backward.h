#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "param_struct.h"
#include "config.h"

namespace BACKWARD
{
	__global__ void preprocessCUDA(
		int W, int H, int P, int D, int M, bool use_shs, bool use_vertex_color, bool rich_info,
		const float *__restrict__ viewmatrix,
		const float *__restrict__ projmatrix,
		const float *__restrict__ campos,
		const float *__restrict__ vertex,
		const float *__restrict__ shs,
		const int *__restrict__ radii,
		const bool *__restrict__ clamped,
		const float3 *__restrict__ dL_dv1_view_ptr,
		const float3 *__restrict__ dL_dv2_view_ptr,
		const float3 *__restrict__ dL_dv3_view_ptr,
		const float *__restrict__ dL_dfeature,
		float *__restrict__ dL_dvertex,
		float *__restrict__ dL_dshs);

	__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
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
			const float *__restrict__ final_Ts,
			const uint32_t *__restrict__ n_contribs,
			const float *__restrict__ depth_image,
			const float *__restrict__ dL_dout_feature,
			const float *__restrict__ dL_dout_depth,
			const float *__restrict__ dL_dout_normal,
			const float *__restrict__ dL_dout_distortion,
			float3 *__restrict__ dL_dv1_view,
			float3 *__restrict__ dL_dv2_view,
			float3 *__restrict__ dL_dv3_view,
			float *__restrict__ dL_dfeature,
			float *__restrict__ dL_dopacity,
			float *__restrict__ dL_dv_norm);

	__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
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
			const float *__restrict__ final_Ts,
			const float *__restrict__ final_features,
			const float *__restrict__ final_normals,
			const float *__restrict__ final_depths,
			const float *__restrict__ final_distorts,
			const float *__restrict__ dL_dout_feature,
			const float *__restrict__ dL_dout_depth,
			const float *__restrict__ dL_dout_normal,
			const float *__restrict__ dL_dout_distortion,
			float3 *__restrict__ dL_dv1_view,
			float3 *__restrict__ dL_dv2_view,
			float3 *__restrict__ dL_dv3_view,
			float *__restrict__ dL_dfeature,
			float *__restrict__ dL_dopacity,
			float *__restrict__ dL_dv_norm);
}
