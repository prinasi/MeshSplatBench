#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "backward.h"
#include "auxiliary.h"

// Backward method for converting the input spherical harmonics coefficients to RGB colors.
__device__ void computeRGBFromSHBackward(int idx, int deg, int max_coeffs, float3 pos, float3 campos, const float *shs, const bool *clamped,
										 const float3 *dL_dfeature, float3 *dL_dshs, float3 &dL_dpos)
{
	// Compute intermediate values, as it is done during forward
	float3 dir_orig = pos - campos;
	float3 dir = dir_orig / norm(dir_orig);

	float3 *sh = ((float3 *)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied, gradient becomes 0.
	float3 dL_dRGB = dL_dfeature[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	float3 dRGBdx = {0, 0, 0};
	float3 dRGBdy = {0, 0, 0};
	float3 dRGBdz = {0, 0, 0};
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location to write SH gradients to
	float3 *dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (SH_C3[0] * sh[9] * 3.f * 2.f * xy +
						   SH_C3[1] * sh[10] * yz +
						   SH_C3[2] * sh[11] * -2.f * xy +
						   SH_C3[3] * sh[12] * -3.f * 2.f * xz +
						   SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
						   SH_C3[5] * sh[14] * 2.f * xz +
						   SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (SH_C3[0] * sh[9] * 3.f * (xx - yy) +
						   SH_C3[1] * sh[10] * xz +
						   SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
						   SH_C3[3] * sh[12] * -3.f * 2.f * yz +
						   SH_C3[4] * sh[13] * -2.f * xy +
						   SH_C3[5] * sh[14] * -2.f * yz +
						   SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (SH_C3[1] * sh[10] * xy +
						   SH_C3[2] * sh[11] * 4.f * 2.f * yz +
						   SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
						   SH_C3[4] * sh[13] * 4.f * 2.f * xz +
						   SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation.
	// View direction is influenced by the triangle center,
	// so SHs gradients must propagate back into 3D position.
	float3 dL_ddir = {dot(dL_dRGB, dRGBdx), dot(dL_dRGB, dRGBdy), dot(dL_dRGB, dRGBdz)};

	// Account for normalization of direction
	dL_dpos += dnormvdv(float3{dir_orig.x, dir_orig.y, dir_orig.z}, float3{dL_ddir.x, dL_ddir.y, dL_ddir.z});
}

__global__ void BACKWARD::preprocessCUDA(
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
	float *__restrict__ dL_dshs)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || radii[idx] <= 0)
		return;

	float3 dL_dv1_view = dL_dv1_view_ptr[idx];
	float3 dL_dv2_view = dL_dv2_view_ptr[idx];
	float3 dL_dv3_view = dL_dv3_view_ptr[idx];

	float3 dL_dv1 = transformVec4x3Transpose(dL_dv1_view, viewmatrix);
	float3 dL_dv2 = transformVec4x3Transpose(dL_dv2_view, viewmatrix);
	float3 dL_dv3 = transformVec4x3Transpose(dL_dv3_view, viewmatrix);

	// Compute gradient updates due to computing RGB from SHs
	if (use_shs)
	{
		const float3 v1 = {vertex[9 * idx + 0], vertex[9 * idx + 1], vertex[9 * idx + 2]};
		const float3 v2 = {vertex[9 * idx + 3], vertex[9 * idx + 4], vertex[9 * idx + 5]};
		const float3 v3 = {vertex[9 * idx + 6], vertex[9 * idx + 7], vertex[9 * idx + 8]};
		if (use_vertex_color)
		{
			computeRGBFromSHBackward(idx * 3 + 0, D, M, v1, *(float3 *)campos, shs, clamped,
									 (const float3 *)dL_dfeature, (float3 *)dL_dshs, dL_dv1);
			computeRGBFromSHBackward(idx * 3 + 1, D, M, v2, *(float3 *)campos, shs, clamped,
									 (const float3 *)dL_dfeature, (float3 *)dL_dshs, dL_dv2);
			computeRGBFromSHBackward(idx * 3 + 2, D, M, v3, *(float3 *)campos, shs, clamped,
									 (const float3 *)dL_dfeature, (float3 *)dL_dshs, dL_dv3);
		}
		else
		{
			const float3 center = (v1 + v2 + v3) / 3.0f;
			float3 dL_dcenter_sh = {0, 0, 0};
			computeRGBFromSHBackward(idx, D, M, center, *(float3 *)campos, shs, clamped,
									 (const float3 *)dL_dfeature, (float3 *)dL_dshs, dL_dcenter_sh);
			dL_dv1 += dL_dcenter_sh / 3.0f;
			dL_dv2 += dL_dcenter_sh / 3.0f;
			dL_dv3 += dL_dcenter_sh / 3.0f;
		}
	}

	dL_dvertex[9 * idx + 0] = dL_dv1.x;
	dL_dvertex[9 * idx + 1] = dL_dv1.y;
	dL_dvertex[9 * idx + 2] = dL_dv1.z;
	dL_dvertex[9 * idx + 3] = dL_dv2.x;
	dL_dvertex[9 * idx + 4] = dL_dv2.y;
	dL_dvertex[9 * idx + 5] = dL_dv2.z;
	dL_dvertex[9 * idx + 6] = dL_dv3.x;
	dL_dvertex[9 * idx + 7] = dL_dv3.y;
	dL_dvertex[9 * idx + 8] = dL_dv3.z;
}

__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	BACKWARD::renderCUDA(
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
		float *__restrict__ dL_dv_norm)
{
	auto block = cg::this_thread_block();
	dim3 group_index = block.group_index();
	dim3 thread_index = block.thread_index();
	auto tid = block.thread_rank();

	const uint2 pix = {group_index.x * BLOCK_X + thread_index.x, group_index.y * BLOCK_Y + thread_index.y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float3 p_ray = {tan_fovx * pixToProj((float)pix.x, W), tan_fovy * pixToProj((float)pix.y, H), 1.0f};

	const bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	const uint2 range = ranges[group_index.y * ((W + BLOCK_X - 1) / BLOCK_X) + group_index.x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_v1_view[BLOCK_SIZE];
	__shared__ float3 collected_v2_view[BLOCK_SIZE];
	__shared__ float3 collected_v3_view[BLOCK_SIZE];
	__shared__ float3 collected_normal_view[BLOCK_SIZE];
	__shared__ float collected_opacity[BLOCK_SIZE];

	const float final_T = inside ? final_Ts[pix_id] : 0;
	float T = final_T;
	const uint32_t last_contributor = inside ? n_contribs[pix_id] : 0;
	uint32_t contributor = toDo;

	float accum_feature[MAX_CHANNELS] = {0}; // Accumulated feature from back to front
	float3 accum_normal = {0, 0, 0};		 // Accumulated normal from back to front
	float accum_depth = background_depth;	 // Accumulated depth from back to front
	float accum_distort = 0;				 // Accumulated distortion from back to front

	float dL_dfeature_pixel[MAX_CHANNELS] = {0};
	float3 dL_dnormal_pixel = {0, 0, 0};
	float dL_ddepth_pixel = 0;
	float dL_ddistortion_pixel = 0;
	float depth_pixel = 0;

	if (inside)
	{
		for (int i = 0; i < C; i++)
		{
			accum_feature[i] = background[i];
			dL_dfeature_pixel[i] = dL_dout_feature[i * H * W + pix_id];
		}
		if (rich_info)
		{
			dL_dnormal_pixel = make_float3(dL_dout_normal[pix_id], dL_dout_normal[W * H + pix_id], dL_dout_normal[2 * W * H + pix_id]);
			dL_ddepth_pixel = dL_dout_depth[pix_id];
			dL_ddistortion_pixel = dL_dout_distortion[pix_id] * (1.0f - final_T);
			depth_pixel = (depth_image[pix_id] - final_T * background_depth) / (1.0f - final_T + EPS);
		}
	}

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory,
		// start from the BACK and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + tid;
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[tid] = coll_id;
			collected_v1_view[tid] = s_v1_view[coll_id];
			collected_v2_view[tid] = s_v2_view[coll_id];
			collected_v3_view[tid] = s_v3_view[coll_id];
			collected_normal_view[tid] = s_normal_view[coll_id];
			collected_opacity[tid] = opacity[coll_id];
		}
		block.sync();

		// Iterate over triangles in the batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current triangle ID.
			// Skip if this one is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float3 v1_view = collected_v1_view[j];
			const float3 v2_view = collected_v2_view[j];
			const float3 v3_view = collected_v3_view[j];
			const float3 normal_view = collected_normal_view[j];

			const float p_ray_dot_n = dot(p_ray, normal_view);
			if (abs(p_ray_dot_n) < EPS)
				continue;
			const float inv_p_ray_dot_n = 1.0f / p_ray_dot_n;
			const float depth = dot(v1_view, normal_view) * inv_p_ray_dot_n;
			if (depth < 0.0f)
				continue;

			const float3 p_view = depth * p_ray;
			const float3 p_v1 = v1_view - p_view;
			const float3 p_v2 = v2_view - p_view;
			const float3 p_v3 = v3_view - p_view;

			const float inv_n_dot_n = 1.0f / dot(normal_view, normal_view);
			const float a1 = dot(cross(p_v2, p_v3), normal_view) * inv_n_dot_n;
			const float a2 = dot(cross(p_v3, p_v1), normal_view) * inv_n_dot_n;
			const float a3 = 1.0f - a1 - a2;
			const float ecc = 1.0f - 3.0f * min(min(a1, a2), a3);
			if (ecc < 0.0f || ecc > 10.0f)
				continue;

			const float power = -0.5f * pow(ecc, 2.0f * gamma);
			const float op = collected_opacity[j];
			const float G = exp(power);
			const float alpha = min(ALPHA_THRES, op * G);
			if (G < G_THRES)
				continue;

			T /= (1.0f - alpha);
			const float contrib = alpha * T;

			// Propagate gradients to per-triangle feature and alpha
			float dL_dcontrib = 0.0f;
			float3 dL_dnormal = {0, 0, 0};
			float dL_ddepth = 0.0f;
			float3 dL_da = {0, 0, 0};

			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float dL_dfeat = dL_dfeature_pixel[ch];
				float feat;
				if (use_vertex_color)
				{
					const float feat1 = feature[global_id * 3 * C + 0 * C + ch];
					const float feat2 = feature[global_id * 3 * C + 1 * C + ch];
					const float feat3 = feature[global_id * 3 * C + 2 * C + ch];
					feat = feat1 * a1 + feat2 * a2 + feat3 * a3;
					atomicAdd(&(dL_dfeature[global_id * 3 * C + 0 * C + ch]), dL_dfeat * a1 * contrib);
					atomicAdd(&(dL_dfeature[global_id * 3 * C + 1 * C + ch]), dL_dfeat * a2 * contrib);
					atomicAdd(&(dL_dfeature[global_id * 3 * C + 2 * C + ch]), dL_dfeat * a3 * contrib);
					dL_da.x += dL_dfeat * contrib * feat1;
					dL_da.y += dL_dfeat * contrib * feat2;
					dL_da.z += dL_dfeat * contrib * feat3;
				}
				else
				{
					atomicAdd(&(dL_dfeature[global_id * C + ch]), dL_dfeat * contrib);
					feat = feature[global_id * C + ch];
				}
				dL_dcontrib += dL_dfeat * (feat - accum_feature[ch]);
				accum_feature[ch] = alpha * feat + (1.0f - alpha) * accum_feature[ch];
			}

			if (rich_info)
			{
				const float3 normalized_normal_view = normal_view * sqrt(inv_n_dot_n);
				dL_dnormal += dnormvdv(normal_view, dL_dnormal_pixel * contrib);
				dL_dcontrib += dot(dL_dnormal_pixel, normalized_normal_view - accum_normal);
				accum_normal = alpha * normalized_normal_view + (1.0f - alpha) * accum_normal;

				dL_ddepth += dL_ddepth_pixel * contrib;
				dL_dcontrib += dL_ddepth_pixel * (depth - accum_depth);
				accum_depth = alpha * depth + (1.0f - alpha) * accum_depth;

				const float distort = (depth - depth_pixel) * (depth - depth_pixel);
				dL_ddepth += dL_ddistortion_pixel * contrib * 2.0f * (depth - depth_pixel);
				dL_dcontrib += dL_ddistortion_pixel * (distort - accum_distort);
				accum_distort = alpha * distort + (1.0f - alpha) * accum_distort;
			}

			const float dL_dalpha = dL_dcontrib * T;
			const float dL_dpower = (op * G < ALPHA_THRES) ? (dL_dalpha * alpha) : 0.0f; // if not clamped by min(0.99, op * G)
			const float dL_decc = dL_dpower * 2 * gamma * power / (ecc + EPS);

			float3 decc_da = {0, 0, 0};
			if (a1 <= a2 && a1 <= a3)
			{
				decc_da.x = -3.0f;
			}
			else if (a2 <= a1 && a2 <= a3)
			{
				decc_da.y = -3.0f;
			}
			else
			{
				decc_da.z = -3.0f;
			}
			dL_da += dL_decc * decc_da;

			const float3 da1_dv1_view = {0, 0, 0};
			const float3 da1_dv2_view = cross(p_v3, normal_view) * inv_n_dot_n;
			const float3 da1_dv3_view = cross(normal_view, p_v2) * inv_n_dot_n;
			const float3 da1_dnormal_view = (cross(p_v2, p_v3) - 2.0f * a1 * normal_view) * inv_n_dot_n;
			const float da1_ddepth = dot(normal_view, cross(v3_view - v2_view, p_ray)) * inv_n_dot_n;

			const float3 da2_dv1_view = cross(normal_view, p_v3) * inv_n_dot_n;
			const float3 da2_dv2_view = {0, 0, 0};
			const float3 da2_dv3_view = cross(p_v1, normal_view) * inv_n_dot_n;
			const float3 da2_dnormal_view = (cross(p_v3, p_v1) - 2.0f * a2 * normal_view) * inv_n_dot_n;
			const float da2_ddepth = dot(normal_view, cross(v1_view - v3_view, p_ray)) * inv_n_dot_n;

			const float3 da3_dv1_view = -da1_dv1_view - da2_dv1_view;
			const float3 da3_dv2_view = -da1_dv2_view - da2_dv2_view;
			const float3 da3_dv3_view = -da1_dv3_view - da2_dv3_view;
			const float3 da3_dnormal_view = -da1_dnormal_view - da2_dnormal_view;
			const float da3_ddepth = -da1_ddepth - da2_ddepth;

			dL_ddepth += dL_da.x * da1_ddepth + dL_da.y * da2_ddepth + dL_da.z * da3_ddepth;
			const float3 ddepth_dv1_view = normal_view * inv_p_ray_dot_n;
			const float3 ddepth_dnormal_view = p_v1 * inv_p_ray_dot_n;

			float3 dL_dv1_view_point = dL_da.x * da1_dv1_view + dL_da.y * da2_dv1_view + dL_da.z * da3_dv1_view + dL_ddepth * ddepth_dv1_view;
			float3 dL_dv2_view_point = dL_da.x * da1_dv2_view + dL_da.y * da2_dv2_view + dL_da.z * da3_dv2_view;
			float3 dL_dv3_view_point = dL_da.x * da1_dv3_view + dL_da.y * da2_dv3_view + dL_da.z * da3_dv3_view;
			dL_dnormal += dL_da.x * da1_dnormal_view + dL_da.y * da2_dnormal_view + dL_da.z * da3_dnormal_view + dL_ddepth * ddepth_dnormal_view;

			dL_dv1_view_point += cross(v2_view - v3_view, dL_dnormal);
			dL_dv2_view_point += cross(v3_view - v1_view, dL_dnormal);
			dL_dv3_view_point += cross(v1_view - v2_view, dL_dnormal);

			// Update gradients w.r.t. triangle vertex positions
			float *dL_dv1_view_ptr = (float *)&dL_dv1_view[global_id];
			atomicAdd(dL_dv1_view_ptr++, dL_dv1_view_point.x);
			atomicAdd(dL_dv1_view_ptr++, dL_dv1_view_point.y);
			atomicAdd(dL_dv1_view_ptr++, dL_dv1_view_point.z);

			float *dL_dv2_view_ptr = (float *)&dL_dv2_view[global_id];
			atomicAdd(dL_dv2_view_ptr++, dL_dv2_view_point.x);
			atomicAdd(dL_dv2_view_ptr++, dL_dv2_view_point.y);
			atomicAdd(dL_dv2_view_ptr++, dL_dv2_view_point.z);

			float *dL_dv3_view_ptr = (float *)&dL_dv3_view[global_id];
			atomicAdd(dL_dv3_view_ptr++, dL_dv3_view_point.x);
			atomicAdd(dL_dv3_view_ptr++, dL_dv3_view_point.y);
			atomicAdd(dL_dv3_view_ptr++, dL_dv3_view_point.z);

			// Update gradients for densification
			const float dL_dv_norm_point = (norm(dL_dv1_view_point) * v1_view.z + norm(dL_dv2_view_point) * v2_view.z + norm(dL_dv3_view_point) * v3_view.z) / 3.0f;
			atomicAdd(&dL_dv_norm[global_id], dL_dv_norm_point);

			// Update gradients w.r.t. triangle opacity
			atomicAdd(&dL_dopacity[global_id], dL_dalpha * G);
		}
	}
}

struct BlendInfo
{
	int global_id;
	float depth;

	__device__ BlendInfo() : global_id(-1), depth(FLT_MAX) {}
	__device__ BlendInfo(int global_id, float depth) : global_id(global_id), depth(depth) {}
};

__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	BACKWARD::renderCUDAResort(
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
		float *__restrict__ dL_dv_norm)
{
	// Identify current tile and pixel.
	auto block = cg::this_thread_block();
	dim3 group_index = block.group_index();
	dim3 thread_index = block.thread_index();
	auto tid = block.thread_rank();

	const uint2 pix = {group_index.x * BLOCK_X + thread_index.x, group_index.y * BLOCK_Y + thread_index.y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float3 p_ray = {tan_fovx * pixToProj((float)pix.x, W), tan_fovy * pixToProj((float)pix.y, H), 1.0f};

	const bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	const uint2 range = ranges[group_index.y * ((W + BLOCK_X - 1) / BLOCK_X) + group_index.x];

	// Initialize blending variables
	float T = 1.0f;
	float accum_feature[MAX_CHANNELS] = {0}; // Accumulated feature from front to back
	float3 accum_normal = {0, 0, 0};		 // Accumulated normal from front to back
	float accum_depth = 0;					 // Accumulated depth from front to back
	float accum_distort = 0;				 // Accumulated distortion from front to back

	// Final outputs
	const float final_T = inside ? final_Ts[pix_id] : 0;
	float final_feature[MAX_CHANNELS] = {0}; // Final color for this pixel
	float3 final_normal = {0, 0, 0};		 // Final normal for this pixel
	float final_depth = 0;					 // Final depth for this pixel
	float final_depth_accum = 0;			 // Final depth for this pixel (without background)
	float final_distort = 0;				 // Final distortion for this pixel

	// Gradients of loss w.r.t. pixel outputs
	float dL_dfeature_pixel[MAX_CHANNELS] = {0};
	float3 dL_dnormal_pixel = {0, 0, 0};
	float dL_ddepth_pixel = 0;
	float dL_ddistortion_pixel = 0;

	if (inside)
	{
		for (int i = 0; i < C; i++)
		{
			final_feature[i] = final_features[i * H * W + pix_id];
			dL_dfeature_pixel[i] = dL_dout_feature[i * H * W + pix_id];
		}
		if (rich_info)
		{
			final_depth = final_depths[pix_id];
			final_depth_accum = (final_depth - final_T * background_depth) / (1.0f - final_T + EPS);
			final_normal = make_float3(final_normals[pix_id], final_normals[W * H + pix_id], final_normals[2 * W * H + pix_id]);
			final_distort = final_distorts[pix_id];
			dL_dnormal_pixel = make_float3(dL_dout_normal[pix_id], dL_dout_normal[W * H + pix_id], dL_dout_normal[2 * W * H + pix_id]);
			dL_ddepth_pixel = dL_dout_depth[pix_id];
			dL_ddistortion_pixel = dL_dout_distortion[pix_id] * (1.0f - final_T);
		}
	}

	// Storage for sorting a small window of samples
	BlendInfo sort_buffer[SORT_WINDOW_SIZE];
	int sort_num = 0;

	// Blending function that blends one sample at a time
	auto blend_one = [&]()
	{
		if (sort_num == 0)
			return;

		// Always blend the closest sample
		const int global_id = sort_buffer[0].global_id;
		const float depth = sort_buffer[0].depth;

		// Redo some computations because memory allocation is the bottleneck
		const float3 v1_view = s_v1_view[global_id];
		const float3 v2_view = s_v2_view[global_id];
		const float3 v3_view = s_v3_view[global_id];
		const float3 normal_view = s_normal_view[global_id];

		const float3 p_view = depth * p_ray;
		const float3 p_v1 = v1_view - p_view;
		const float3 p_v2 = v2_view - p_view;
		const float3 p_v3 = v3_view - p_view;

		const float inv_n_dot_n = 1.0f / dot(normal_view, normal_view);
		const float a1 = dot(cross(p_v2, p_v3), normal_view) * inv_n_dot_n;
		const float a2 = dot(cross(p_v3, p_v1), normal_view) * inv_n_dot_n;
		const float a3 = 1.0f - a1 - a2;
		const float ecc = 1.0f - 3.0f * min(min(a1, a2), a3);

		const float power = -0.5f * pow(ecc, 2.0f * gamma);
		const float op = opacity[global_id];
		const float G = exp(power);
		const float alpha = min(ALPHA_THRES, op * G);

		const float contrib = alpha * T;
		const float test_T = T * (1.0f - alpha);

		const float dalpha_dpower = (op * G < ALPHA_THRES) ? alpha : 0.0f; // if not clamped by min(0.99, op * G)
		const float dalpha_decc = dalpha_dpower * 2 * gamma * power / (ecc + EPS);

		// Propagate gradients to per-triangle parameters
		float dL_dcontrib = 0.0f;
		float3 dL_dnormal = {0, 0, 0};
		float dL_ddepth = 0.0f;
		float3 dL_da = {0, 0, 0};

		for (int ch = 0; ch < C; ch++)
		{
			const float dL_dfeat = dL_dfeature_pixel[ch];
			float feat;
			if (use_vertex_color)
			{
				const float feat1 = feature[global_id * 3 * C + 0 * C + ch];
				const float feat2 = feature[global_id * 3 * C + 1 * C + ch];
				const float feat3 = feature[global_id * 3 * C + 2 * C + ch];
				feat = feat1 * a1 + feat2 * a2 + feat3 * a3;
				atomicAdd(&(dL_dfeature[global_id * 3 * C + 0 * C + ch]), dL_dfeat * a1 * contrib);
				atomicAdd(&(dL_dfeature[global_id * 3 * C + 1 * C + ch]), dL_dfeat * a2 * contrib);
				atomicAdd(&(dL_dfeature[global_id * 3 * C + 2 * C + ch]), dL_dfeat * a3 * contrib);
				dL_da.x += dL_dfeat * contrib * feat1;
				dL_da.y += dL_dfeat * contrib * feat2;
				dL_da.z += dL_dfeat * contrib * feat3;
			}
			else
			{
				atomicAdd(&(dL_dfeature[global_id * C + ch]), dL_dfeat * contrib);
				feat = feature[global_id * C + ch];
			}

			accum_feature[ch] += feat * contrib;
			const float accum_feat_back = (final_feature[ch] - accum_feature[ch]) / test_T; // accumulated feature coming after
			dL_dcontrib += dL_dfeat * (feat - accum_feat_back);
		}

		if (rich_info)
		{
			const float3 normal = normal_view * sqrt(inv_n_dot_n); // normalized normal
			
			accum_normal += normal * contrib;
			accum_depth += depth * contrib;
			const float distort = (depth - final_depth_accum) * (depth - final_depth_accum);
			accum_distort += distort * contrib;

			const float3 accum_normal_back = (final_normal - accum_normal) / test_T;   // accumulated normal coming after
			const float accum_depth_back = (final_depth - accum_depth) / test_T;	   // accumulated depth coming after
			const float accum_distort_back = (final_distort - accum_distort) / test_T; // accumulated distortion coming after

			dL_dnormal += dnormvdv(normal_view, dL_dnormal_pixel * contrib); // dnormvdv expects unnormalized input
			dL_dcontrib += dot(dL_dnormal_pixel, normal - accum_normal_back);

			dL_ddepth += dL_ddepth_pixel * contrib;
			dL_dcontrib += dL_ddepth_pixel * (depth - accum_depth_back);

			dL_ddepth += dL_ddistortion_pixel * contrib * 2.0f * (depth - final_depth_accum);
			dL_dcontrib += dL_ddistortion_pixel * (distort - accum_distort_back);
		}

		const float dL_dalpha = dL_dcontrib * T;
		const float dL_decc = dL_dalpha * dalpha_decc;
		const float inv_p_ray_dot_n = 1.0f / dot(p_ray, normal_view);

		float3 decc_da = {0, 0, 0};
		if (a1 <= a2 && a1 <= a3)
		{
			decc_da.x = -3.0f;
		}
		else if (a2 <= a1 && a2 <= a3)
		{
			decc_da.y = -3.0f;
		}
		else
		{
			decc_da.z = -3.0f;
		}
		dL_da += dL_decc * decc_da;

		const float3 da1_dv1_view = {0, 0, 0};
		const float3 da1_dv2_view = cross(p_v3, normal_view) * inv_n_dot_n;
		const float3 da1_dv3_view = cross(normal_view, p_v2) * inv_n_dot_n;
		const float3 da1_dnormal_view = (cross(p_v2, p_v3) - 2.0f * a1 * normal_view) * inv_n_dot_n;
		const float da1_ddepth = dot(normal_view, cross(v3_view - v2_view, p_ray)) * inv_n_dot_n;

		const float3 da2_dv1_view = cross(normal_view, p_v3) * inv_n_dot_n;
		const float3 da2_dv2_view = {0, 0, 0};
		const float3 da2_dv3_view = cross(p_v1, normal_view) * inv_n_dot_n;
		const float3 da2_dnormal_view = (cross(p_v3, p_v1) - 2.0f * a2 * normal_view) * inv_n_dot_n;
		const float da2_ddepth = dot(normal_view, cross(v1_view - v3_view, p_ray)) * inv_n_dot_n;

		// const float3 da3_dv1_view = cross(p_v2, normal_view) * inv_n_dot_n;
		// const float3 da3_dv2_view = cross(normal_view, p_v1) * inv_n_dot_n;
		// const float3 da3_dv3_view = {0, 0, 0};
		// const float3 da3_dnormal_view = (cross(p_v1, p_v2) - 2.0f * a3 * normal_view) * inv_n_dot_n;
		// const float da3_ddepth = dot(normal_view, cross(v2_view - v1_view, p_ray)) * inv_n_dot_n;

		const float3 da3_dv1_view = -da1_dv1_view - da2_dv1_view;
		const float3 da3_dv2_view = -da1_dv2_view - da2_dv2_view;
		const float3 da3_dv3_view = -da1_dv3_view - da2_dv3_view;
		const float3 da3_dnormal_view = -da1_dnormal_view - da2_dnormal_view;
		const float da3_ddepth = -da1_ddepth - da2_ddepth;

		dL_ddepth += dL_da.x * da1_ddepth + dL_da.y * da2_ddepth + dL_da.z * da3_ddepth;
		const float3 ddepth_dv1_view = normal_view * inv_p_ray_dot_n;
		const float3 ddepth_dnormal_view = p_v1 * inv_p_ray_dot_n;

		float3 dL_dv1_view_point = dL_da.x * da1_dv1_view + dL_da.y * da2_dv1_view + dL_da.z * da3_dv1_view + dL_ddepth * ddepth_dv1_view;
		float3 dL_dv2_view_point = dL_da.x * da1_dv2_view + dL_da.y * da2_dv2_view + dL_da.z * da3_dv2_view;
		float3 dL_dv3_view_point = dL_da.x * da1_dv3_view + dL_da.y * da2_dv3_view + dL_da.z * da3_dv3_view;
		dL_dnormal += dL_da.x * da1_dnormal_view + dL_da.y * da2_dnormal_view + dL_da.z * da3_dnormal_view + dL_ddepth * ddepth_dnormal_view;

		dL_dv1_view_point += cross(v2_view - v3_view, dL_dnormal);
		dL_dv2_view_point += cross(v3_view - v1_view, dL_dnormal);
		dL_dv3_view_point += cross(v1_view - v2_view, dL_dnormal);

		// Update gradients w.r.t. triangle vertex positions
		float *dL_dv1_view_ptr = (float *)&dL_dv1_view[global_id];
		atomicAdd(dL_dv1_view_ptr++, dL_dv1_view_point.x);
		atomicAdd(dL_dv1_view_ptr++, dL_dv1_view_point.y);
		atomicAdd(dL_dv1_view_ptr++, dL_dv1_view_point.z);

		float *dL_dv2_view_ptr = (float *)&dL_dv2_view[global_id];
		atomicAdd(dL_dv2_view_ptr++, dL_dv2_view_point.x);
		atomicAdd(dL_dv2_view_ptr++, dL_dv2_view_point.y);
		atomicAdd(dL_dv2_view_ptr++, dL_dv2_view_point.z);

		float *dL_dv3_view_ptr = (float *)&dL_dv3_view[global_id];
		atomicAdd(dL_dv3_view_ptr++, dL_dv3_view_point.x);
		atomicAdd(dL_dv3_view_ptr++, dL_dv3_view_point.y);
		atomicAdd(dL_dv3_view_ptr++, dL_dv3_view_point.z);

		// Update gradients for densification
		const float dL_dv_norm_point = (norm(dL_dv1_view_point) * v1_view.z + norm(dL_dv2_view_point) * v2_view.z + norm(dL_dv3_view_point) * v3_view.z) / 3.0f;
		atomicAdd(&dL_dv_norm[global_id], dL_dv_norm_point);

		// Update gradients w.r.t. triangle opacity
		atomicAdd(&dL_dopacity[global_id], dL_dalpha * G);

		T *= (1.0f - alpha);
		if (T <= T_THRES)
			done = true;

		// Pop the first element
		for (int i = 1; i < SORT_WINDOW_SIZE && sort_buffer[i - 1].depth != FLT_MAX; ++i)
		{
			sort_buffer[i - 1] = sort_buffer[i];
		}
		sort_buffer[SORT_WINDOW_SIZE - 1].depth = FLT_MAX;
		--sort_num;
	};

	// Iterate over samples until done or range is complete
	for (int i = range.x; !done && i < range.y; i++)
	{
		// Find the next sample to blend
		const int global_id = point_list[i];
		const float3 v1_view = s_v1_view[global_id];
		const float3 v2_view = s_v2_view[global_id];
		const float3 v3_view = s_v3_view[global_id];
		const float3 normal_view = s_normal_view[global_id];

		const float p_ray_dot_n = dot(p_ray, normal_view);
		if (abs(p_ray_dot_n) < EPS)
			continue;
		const float depth = dot(v1_view, normal_view) / p_ray_dot_n;
		if (depth < 0.0f)
			continue;

		const float3 p_view = depth * p_ray;
		const float3 p_v1 = v1_view - p_view;
		const float3 p_v2 = v2_view - p_view;
		const float3 p_v3 = v3_view - p_view;

		const float inv_n_dot_n = 1.0f / dot(normal_view, normal_view);
		const float a1 = dot(cross(p_v2, p_v3), normal_view) * inv_n_dot_n;
		const float a2 = dot(cross(p_v3, p_v1), normal_view) * inv_n_dot_n;
		const float a3 = 1.0f - a1 - a2;
		const float ecc = 1.0f - 3.0f * min(min(a1, a2), a3);
		if (ecc < 0.0f || ecc > 10.0f)
			continue;

		const float power = -0.5f * pow(ecc, 2.0f * gamma);
		const float op = opacity[global_id];
		const float G = exp(power);
		const float alpha = min(ALPHA_THRES, op * G);
		if (G < G_THRES)
			continue;

		// Push new sample into the sort buffer
		BlendInfo new_sample(global_id, depth);
		for (int s = 0; s < SORT_WINDOW_SIZE && new_sample.depth != FLT_MAX; ++s)
		{
			if (new_sample.depth < sort_buffer[s].depth)
			{
				swap(new_sample, sort_buffer[s]);
			}
		}
		++sort_num;

		// Blend one sample if the buffer is full
		if (sort_num == SORT_WINDOW_SIZE)
			blend_one();
	}

	// Blend remaining samples in the buffer
	while (!done && sort_num > 0)
		blend_one();
}
