#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "rasterizer.h"
#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

__forceinline__ __device__ float getEcc(const float3 &v1_view, const float3 &v2_view, const float3 &v3_view, const float3 &normal_view, const float3 &p_ray, float &depth)
{
	float p_ray_dot_n = dot(p_ray, normal_view);
	p_ray_dot_n = p_ray_dot_n > 0 ? max(p_ray_dot_n, EPS) : min(p_ray_dot_n, -EPS);
	depth = dot(v1_view, normal_view) / p_ray_dot_n;

	const float3 p_view = depth * p_ray;
	const float3 p_v1 = v1_view - p_view;
	const float3 p_v2 = v2_view - p_view;
	const float3 p_v3 = v3_view - p_view;

	const float inv_n_dot_n = 1.0f / dot(normal_view, normal_view);
	const float a1 = dot(cross(p_v2, p_v3), normal_view) * inv_n_dot_n;
	const float a2 = dot(cross(p_v3, p_v1), normal_view) * inv_n_dot_n;
	const float a3 = 1.0f - a1 - a2;
	const float ecc = 1.0f - 3.0f * min(min(a1, a2), a3);
	return ecc;
}

__forceinline__ __device__ void testRay(
	const float3 &v1_view,
	const float3 &v2_view,
	const float3 &v3_view,
	const float3 &normal_view,
	const float3 &p_ray,
	float &lowest_ecc,
	float &depth)
{
	float temp_depth;
	const float ecc = getEcc(v1_view, v2_view, v3_view, normal_view, p_ray, temp_depth);
	if (ecc < lowest_ecc)
	{
		lowest_ecc = ecc;
		depth = temp_depth;
	}
}

__forceinline__ __device__ bool findIntersection(const float2 &a1, const float2 &a2, const float2 &b1, const float2 &b2, float2 &intersection)
{
	// a1 -> a2 is line ray
	// b1 -> b2 is line segment
	float denom = (b2.y - b1.y) * (a2.x - a1.x) - (b2.x - b1.x) * (a2.y - a1.y); // (a2 - a1) x (b2 - b1)
	if (denom == 0.0f)
		return false; // Lines are parallel

	float ua = ((b2.x - b1.x) * (a1.y - b1.y) - (b2.y - b1.y) * (a1.x - b1.x)) / denom; // (b2 - b1) x (a1 - b1)
	float ub = ((a2.x - a1.x) * (a1.y - b1.y) - (a2.y - a1.y) * (a1.x - b1.x)) / denom; // (a2 - a1) x (a1 - b1)

	if (ua < 0.0f || ub < 0.0f || ub > 1.0f)
		return false; // Intersection not within line segments

	intersection = a1 + ua * (a2 - a1);
	return true;
}

__forceinline__ __device__ void testEdge(
	const float2 &v1_2D,
	const float2 &v2_2D,
	const float2 &v3_2D,
	const float2 &center_2D,
	const float2 &edge_start,
	const float2 &edge_end,
	const float3 &v1_view,
	const float3 &v2_view,
	const float3 &v3_view,
	const float3 &normal_view,
	int W, int H, float tan_fovx, float tan_fovy,
	float &lowest_ecc,
	float &depth)
{
	// Find intersection between rays from center_2D to each vertex with the edge and testRay at the intersection
	float2 intersection;
	float3 p_ray;
	if (findIntersection(center_2D, v1_2D, edge_start, edge_end, intersection))
	{
		p_ray = make_float3(tan_fovx * pixToProj(intersection.x, W), tan_fovy * pixToProj(intersection.y, H), 1.0f);
		testRay(v1_view, v2_view, v3_view, normal_view, p_ray, lowest_ecc, depth);
	}
	if (findIntersection(center_2D, v2_2D, edge_start, edge_end, intersection))
	{
		p_ray = make_float3(tan_fovx * pixToProj(intersection.x, W), tan_fovy * pixToProj(intersection.y, H), 1.0f);
		testRay(v1_view, v2_view, v3_view, normal_view, p_ray, lowest_ecc, depth);
	}
	if (findIntersection(center_2D, v3_2D, edge_start, edge_end, intersection))
	{
		p_ray = make_float3(tan_fovx * pixToProj(intersection.x, W), tan_fovy * pixToProj(intersection.y, H), 1.0f);
		testRay(v1_view, v2_view, v3_view, normal_view, p_ray, lowest_ecc, depth);
	}
}

__device__ float getLowestEcc(
	int W, int H, float tan_fovx, float tan_fovy,
	const float3 &v1_view,
	const float3 &v2_view,
	const float3 &v3_view,
	const float3 &normal_view,
	const float2 &v1_2D,
	const float2 &v2_2D,
	const float2 &v3_2D,
	const float2 &center_2D,
	const float2 &bbox_min,
	const float2 &bbox_max,
	float &depth)
{
	float lowest_ecc = FLT_MAX;
	const float2 bbox_min_view = {tan_fovx * pixToProj(bbox_min.x, W), tan_fovy * pixToProj(bbox_min.y, H)};
	const float2 bbox_max_view = {tan_fovx * pixToProj(bbox_max.x, W), tan_fovy * pixToProj(bbox_max.y, H)};

	/*
	There are 9 cases where the center of the triangle lies relative to the bbox:
	1 │ 2 │ 3
	──┼───┼──
	4 │ 0 │ 5
	──┼───┼──
	6 │ 7 │ 8
	*/
	if (center_2D.x >= bbox_min.x && center_2D.x <= bbox_max.x && center_2D.y >= bbox_min.y && center_2D.y <= bbox_max.y) // case 0
	{
		lowest_ecc = 0.0f;
	}
	else if (center_2D.x < bbox_min.x && center_2D.y < bbox_min.y) // case 1
	{
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
		testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_min.y), make_float2(bbox_min.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
		testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_min.y), make_float2(bbox_max.x, bbox_min.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
	}
	else if (center_2D.x >= bbox_min.x && center_2D.x <= bbox_max.x && center_2D.y < bbox_min.y) // case 2
	{
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
		testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_min.y), make_float2(bbox_max.x, bbox_min.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
	}
	else if (center_2D.x > bbox_max.x && center_2D.y < bbox_min.y) // case 3
	{
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
		testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_min.y), make_float2(bbox_max.x, bbox_min.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
		testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_max.x, bbox_min.y), make_float2(bbox_max.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
	}
	else if (center_2D.x < bbox_min.x && center_2D.y >= bbox_min.y && center_2D.y <= bbox_max.y) // case 4
	{
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
		testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_min.y), make_float2(bbox_min.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
	}
	else if (center_2D.x > bbox_max.x && center_2D.y >= bbox_min.y && center_2D.y <= bbox_max.y) // case 5
	{
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
		testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_max.x, bbox_min.y), make_float2(bbox_max.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
	}
	else if (center_2D.x < bbox_min.x && center_2D.y > bbox_max.y) // case 6
	{
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
		testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_min.y), make_float2(bbox_min.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
		testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_max.y), make_float2(bbox_max.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
	}
	else if (center_2D.x >= bbox_min.x && center_2D.x <= bbox_max.x && center_2D.y > bbox_max.y) // case 7
	{
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
		testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_max.y), make_float2(bbox_max.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
	}
	else if (center_2D.x > bbox_max.x && center_2D.y > bbox_max.y) // case 8
	{
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
		testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
		testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_max.x, bbox_min.y), make_float2(bbox_max.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
		testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_max.y), make_float2(bbox_max.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
	}

	return lowest_ecc;
}

__global__ void duplicateWithKeysPerTileDepth(
	int P, dim3 grid, int W, int H, float tan_fovx, float tan_fovy, float gamma,
	const uint32_t *tiles_touched,
	const uint2 *rect_min,
	const uint2 *rect_max,
	const float *depth,
	const uint32_t *offsets,
	const float3 *__restrict__ s_v1_view,
	const float3 *__restrict__ s_v2_view,
	const float3 *__restrict__ s_v3_view,
	const float3 *__restrict__ s_normal_view,
	const float *__restrict__ projmatrix,
	uint64_t *point_list_keys_unsorted,
	uint32_t *point_list_values_unsorted)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	if (tiles_touched[idx] <= 0)
		return;

	uint2 cur_rect_min = rect_min[idx];
	uint2 cur_rect_max = rect_max[idx];
	float cur_depth = depth[idx];
	uint32_t cur_offset = (idx == 0) ? 0 : offsets[idx - 1];

	const float3 v1_view = s_v1_view[idx];
	const float3 v2_view = s_v2_view[idx];
	const float3 v3_view = s_v3_view[idx];
	const float3 normal_view = s_normal_view[idx];
	const float3 center_view = (v1_view + v2_view + v3_view) / 3.0f;

	const float3 v1_view_clip = make_float3(v1_view.x, v1_view.y, max(v1_view.z, 0.001f));
	const float3 v2_view_clip = make_float3(v2_view.x, v2_view.y, max(v2_view.z, 0.001f));
	const float3 v3_view_clip = make_float3(v3_view.x, v3_view.y, max(v3_view.z, 0.001f));
	const float3 center_view_clip = make_float3(center_view.x, center_view.y, max(center_view.z, 0.001f));

	const float3 v1_proj = projectPoint(v1_view_clip, projmatrix);
	const float3 v2_proj = projectPoint(v2_view_clip, projmatrix);
	const float3 v3_proj = projectPoint(v3_view_clip, projmatrix);
	const float3 center_proj = projectPoint(center_view_clip, projmatrix);

	const float2 v1_2D = {projToPix(v1_proj.x, W), projToPix(v1_proj.y, H)};
	const float2 v2_2D = {projToPix(v2_proj.x, W), projToPix(v2_proj.y, H)};
	const float2 v3_2D = {projToPix(v3_proj.x, W), projToPix(v3_proj.y, H)};
	const float2 center_2D = {projToPix(center_proj.x, W), projToPix(center_proj.y, H)};

	const float ecc_thres = pow(-2.0f * log(G_THRES), 0.5f / gamma);

	// For each tile that the bounding rect overlaps, emit a key/value pair.
	// The key is | tile ID | depth |, and the value is the ID of the triangle.
	// Sorting the values with this key yields triangle IDs in a list,
	// such that they are first sorted by tile and then by depth.
	for (int y = cur_rect_min.y; y < cur_rect_max.y; y++)
	{
		for (int x = cur_rect_min.x; x < cur_rect_max.x; x++)
		{
			const float2 bbox_min = make_float2(x * BLOCK_X - 0.5f, y * BLOCK_Y - 0.5f);
			const float2 bbox_max = make_float2(min((x + 1) * BLOCK_X - 0.5f, W - 0.5f), min((y + 1) * BLOCK_Y - 0.5f, H - 0.5f));
			const float lowest_ecc = getLowestEcc(W, H, tan_fovx, tan_fovy, v1_view, v2_view, v3_view, normal_view, v1_2D, v2_2D, v3_2D, center_2D, bbox_min, bbox_max, cur_depth);
			cur_depth = max(cur_depth, 0.0f);

			uint64_t key = lowest_ecc < ecc_thres ? (y * grid.x + x) : (grid.x * grid.y); // use an invalid tile ID to indicate ignored tile
			key <<= 32;
			key |= *((uint32_t *)&cur_depth);
			point_list_keys_unsorted[cur_offset] = key;
			point_list_values_unsorted[cur_offset] = idx;
			cur_offset++;
		}
	}
}

__global__ void duplicateWithKeys(
	int P, dim3 grid,
	const uint32_t *tiles_touched,
	const uint2 *rect_min,
	const uint2 *rect_max,
	const float *depth,
	const uint32_t *offsets,
	uint64_t *point_list_keys_unsorted,
	uint32_t *point_list_values_unsorted)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	if (tiles_touched[idx] <= 0)
		return;

	uint2 cur_rect_min = rect_min[idx];
	uint2 cur_rect_max = rect_max[idx];
	float cur_depth = depth[idx];
	cur_depth = max(cur_depth, 0.0f);
	uint32_t cur_offset = (idx == 0) ? 0 : offsets[idx - 1];

	// For each tile that the bounding rect overlaps, emit a key/value pair.
	// The key is | tile ID | depth |, and the value is the ID of the triangle.
	// Sorting the values with this key yields triangle IDs in a list,
	// such that they are first sorted by tile and then by depth.
	for (int y = cur_rect_min.y; y < cur_rect_max.y; y++)
	{
		for (int x = cur_rect_min.x; x < cur_rect_max.x; x++)
		{
			uint64_t key = y * grid.x + x;
			key <<= 32;
			key |= *((uint32_t *)&cur_depth);
			point_list_keys_unsorted[cur_offset] = key;
			point_list_values_unsorted[cur_offset] = idx;
			cur_offset++;
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in the full sorted list.
// If yes, write start/end of this tile. Run once per instanced (duplicated) triangle ID.
__global__ void identifyTileRanges(int L, uint64_t *point_list_keys, uint2 *ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	uint32_t cur_tile = point_list_keys[idx] >> 32;
	if (idx == 0)
		ranges[cur_tile].x = 0;
	else
	{
		uint32_t prev_tile = point_list_keys[idx - 1] >> 32;
		if (cur_tile != prev_tile)
		{
			ranges[prev_tile].y = idx;
			ranges[cur_tile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[cur_tile].y = L;
}

void Rasterizer::forward(
	const Params::CameraInfo &cameraInfo,
	const Params::GeometryInfo &geometryInfo,
	Params::ForwardOutput &forwardOutput,
	const bool back_culling,
	const bool rich_info,
	const int sort_level,
	const bool debug)
{
	const int W = cameraInfo.width;
	const int H = cameraInfo.height;
	const int P = geometryInfo.P;

	const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	Params::GeometryState geometryState(forwardOutput.geometryBuffer, (size_t)P, true, geometryInfo.use_vertex_color);

	FORWARD::preprocessCUDA<<<(P + 255) / 256, 256>>>(
		W, H, P, geometryInfo.D, geometryInfo.M, geometryInfo.gamma, rich_info,
		geometryInfo.use_shs, geometryInfo.use_vertex_color, back_culling, grid,
		cameraInfo.viewmatrix,
		cameraInfo.projmatrix,
		cameraInfo.campos,
		geometryInfo.vertex,
		geometryInfo.shs,
		geometryInfo.opacity,
		forwardOutput.radii,
		geometryState.v1_view,
		geometryState.v2_view,
		geometryState.v3_view,
		geometryState.normal_view,
		geometryState.depth,
		geometryState.rgb,
		geometryState.clamped,
		geometryState.tiles_touched,
		geometryState.rect_min,
		geometryState.rect_max);
	CHECK_CUDA(debug);

#ifdef DEBUG
	// Debugging: print out triangle properties
	float3 v1, v2, v3;
	float3 v1_view, v2_view, v3_view;
	uint2 rect_min, rect_max;
	int radii;
	for (int i = 0; i < P; i++)
	{
		cudaMemcpy(&v1, geometryInfo.vertex + 9 * i, sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v2, geometryInfo.vertex + 9 * i + 3, sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v3, geometryInfo.vertex + 9 * i + 6, sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v1_view, geometryState.v1_view + i, sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v2_view, geometryState.v2_view + i, sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v3_view, geometryState.v3_view + i, sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(&rect_min, geometryState.rect_min + i, sizeof(uint2), cudaMemcpyDeviceToHost);
		cudaMemcpy(&rect_max, geometryState.rect_max + i, sizeof(uint2), cudaMemcpyDeviceToHost);
		cudaMemcpy(&radii, forwardOutput.radii + i, sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "P: " << i << ", ";
		std::cout << "v1: [" << v1.x << ", " << v1.y << ", " << v1.z << "], ";
		std::cout << "v2: [" << v2.x << ", " << v2.y << ", " << v2.z << "], ";
		std::cout << "v3: [" << v3.x << ", " << v3.y << ", " << v3.z << "], ";
		std::cout << "v1_view: [" << v1_view.x << ", " << v1_view.y << ", " << v1_view.z << "], ";
		std::cout << "v2_view: [" << v2_view.x << ", " << v2_view.y << ", " << v2_view.z << "], ";
		std::cout << "v3_view: [" << v3_view.x << ", " << v3_view.y << ", " << v3_view.z << "], ";
		std::cout << "rect_min: [" << rect_min.x << ", " << rect_min.y << "], ";
		std::cout << "rect_max: [" << rect_max.x << ", " << rect_max.y << "], ";
		std::cout << "radii: " << radii << std::endl;
	}
#endif

	// Compute prefix sum over full list of touched tile counts.
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	cub::DeviceScan::InclusiveSum(geometryState.scanning_space, geometryState.scan_size, geometryState.tiles_touched, geometryState.point_offsets, P);
	CHECK_CUDA(debug);

	// Retrieve total number of triangle instances to launch and resize aux buffers
	int num_rendered;
	cudaMemcpy(&num_rendered, geometryState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);
	CHECK_CUDA(debug);
	forwardOutput.num_rendered = num_rendered;

	Params::BinningState binningState(forwardOutput.binningBuffer, (size_t)num_rendered, true);

	// For each instance to be rendered, produce adequate [ tile | depth ] key
	// and corresponding dublicated triangle indices to be sorted
	if (sort_level == 0)
	{
		duplicateWithKeys<<<(P + 255) / 256, 256>>>(
			P, grid,
			geometryState.tiles_touched,
			geometryState.rect_min,
			geometryState.rect_max,
			geometryState.depth,
			geometryState.point_offsets,
			binningState.point_list_keys_unsorted,
			binningState.point_list_unsorted);
	}
	else
	{
		duplicateWithKeysPerTileDepth<<<(P + 255) / 256, 256>>>(
			P, grid, W, H, cameraInfo.tan_fovx, cameraInfo.tan_fovy, geometryInfo.gamma,
			geometryState.tiles_touched,
			geometryState.rect_min,
			geometryState.rect_max,
			geometryState.depth,
			geometryState.point_offsets,
			geometryState.v1_view,
			geometryState.v2_view,
			geometryState.v3_view,
			geometryState.normal_view,
			cameraInfo.projmatrix,
			binningState.point_list_keys_unsorted,
			binningState.point_list_unsorted);
	}
	CHECK_CUDA(debug);

	int bit = getHigherMsb(grid.x * grid.y);
	cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted,
		binningState.point_list_keys,
		binningState.point_list_unsorted,
		binningState.point_list,
		num_rendered, 0, 32 + bit);
	CHECK_CUDA(debug);

	Params::ImageState imageState(forwardOutput.imageBuffer, (size_t)(W * H), true);

	cudaMemset(imageState.ranges, 0, (grid.x * grid.y + 1) * sizeof(uint2));
	CHECK_CUDA(debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
	{
		identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(num_rendered, binningState.point_list_keys, imageState.ranges);
		CHECK_CUDA(debug);
	}

#ifdef DEBUG
	// Debugging: print out ranges
	uint2 h_ranges;
	for (int i = 0; i < grid.x * grid.y; i++)
	{
		cudaMemcpy(&h_ranges, imageState.ranges + i, sizeof(uint2), cudaMemcpyDeviceToHost);
		std::cout << "Tile " << i << ", start: " << h_ranges.x << ", end: " << h_ranges.y << std::endl;
	}
#endif

	// Let each tile blend its range of Triangles independently in parallel
	const float *feature = geometryInfo.use_shs ? (float *)geometryState.rgb : geometryInfo.feature;
	if (sort_level <= 1)
	{
		FORWARD::renderCUDA<<<grid, block>>>(
			W, H, geometryInfo.C, geometryInfo.gamma, rich_info, geometryInfo.use_vertex_color, back_culling,
			cameraInfo.tan_fovx,
			cameraInfo.tan_fovy,
			imageState.ranges,
			binningState.point_list,
			geometryState.v1_view,
			geometryState.v2_view,
			geometryState.v3_view,
			geometryState.normal_view,
			feature,
			geometryInfo.opacity,
			geometryInfo.background_depth,
			geometryInfo.background,
			imageState.final_Ts,
			imageState.n_contribs,
			forwardOutput.out_feature,
			forwardOutput.depth,
			forwardOutput.normal,
			forwardOutput.distortion,
			forwardOutput.contrib_sum,
			forwardOutput.contrib_max);
	}
	else
	{
		FORWARD::renderCUDAResort<<<grid, block>>>(
			W, H, geometryInfo.C, geometryInfo.gamma, rich_info, geometryInfo.use_vertex_color, back_culling,
			cameraInfo.tan_fovx,
			cameraInfo.tan_fovy,
			imageState.ranges,
			binningState.point_list,
			geometryState.v1_view,
			geometryState.v2_view,
			geometryState.v3_view,
			geometryState.normal_view,
			feature,
			geometryInfo.opacity,
			geometryInfo.background_depth,
			geometryInfo.background,
			imageState.final_Ts,
			imageState.n_contribs,
			forwardOutput.out_feature,
			forwardOutput.depth,
			forwardOutput.normal,
			forwardOutput.distortion,
			forwardOutput.contrib_sum,
			forwardOutput.contrib_max);
	}
	CHECK_CUDA(debug);

	// copy n_contribs and final_Ts to output buffers
	if (rich_info)
	{
		cudaMemcpy(forwardOutput.n_contribs, imageState.n_contribs, W * H * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(forwardOutput.final_Ts, imageState.final_Ts, W * H * sizeof(float), cudaMemcpyDeviceToDevice);
		CHECK_CUDA(debug);
	}
}

void Rasterizer::backward(
	const Params::CameraInfo &cameraInfo,
	const Params::GeometryInfo &geometryInfo,
	const Params::BackwardInput &backwardInput,
	const Params::LossInput &lossInput,
	Params::BackwardOutput &backwardOutput,
	const bool back_culling,
	const bool rich_info,
	const int sort_level,
	const bool debug)
{
	const int W = cameraInfo.width;
	const int H = cameraInfo.height;
	const int P = geometryInfo.P;

	const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	Params::GeometryState geometryState(backwardInput.geometryBuffer, (size_t)P, false, geometryInfo.use_vertex_color);
	Params::BinningState binningState(backwardInput.binningBuffer, (size_t)backwardInput.num_rendered);
	Params::ImageState imageState(backwardInput.imageBuffer, (size_t)(W * H));

	auto float_opts = backwardInput.geometryBuffer.options().dtype(torch::kFloat32);
	torch::Tensor dL_dv1_view = torch::zeros({P, 3}, float_opts);
	torch::Tensor dL_dv2_view = torch::zeros({P, 3}, float_opts);
	torch::Tensor dL_dv3_view = torch::zeros({P, 3}, float_opts);

	float3 *dL_dv1_view_ptr = (float3 *)dL_dv1_view.data_ptr<float>();
	float3 *dL_dv2_view_ptr = (float3 *)dL_dv2_view.data_ptr<float>();
	float3 *dL_dv3_view_ptr = (float3 *)dL_dv3_view.data_ptr<float>();

	const float *feature = geometryInfo.use_shs ? (float *)geometryState.rgb : geometryInfo.feature;
	if (sort_level <= 1)
	{
		BACKWARD::renderCUDA<<<grid, block>>>(
			W, H, geometryInfo.C, geometryInfo.gamma, rich_info, geometryInfo.use_vertex_color, back_culling,
			cameraInfo.tan_fovx,
			cameraInfo.tan_fovy,
			imageState.ranges,
			binningState.point_list,
			geometryState.v1_view,
			geometryState.v2_view,
			geometryState.v3_view,
			geometryState.normal_view,
			feature,
			geometryInfo.opacity,
			geometryInfo.background_depth,
			geometryInfo.background,
			imageState.final_Ts,
			imageState.n_contribs,
			backwardInput.depth,
			lossInput.dL_dout_feature,
			lossInput.dL_dout_depth,
			lossInput.dL_dout_normal,
			lossInput.dL_dout_distortion,
			dL_dv1_view_ptr,
			dL_dv2_view_ptr,
			dL_dv3_view_ptr,
			backwardOutput.dL_dfeature,
			backwardOutput.dL_dopacity,
			backwardOutput.dL_dv_norm);
	}
	else
	{
		BACKWARD::renderCUDAResort<<<grid, block>>>(
			W, H, geometryInfo.C, geometryInfo.gamma, rich_info, geometryInfo.use_vertex_color, back_culling,
			cameraInfo.tan_fovx,
			cameraInfo.tan_fovy,
			imageState.ranges,
			binningState.point_list,
			geometryState.v1_view,
			geometryState.v2_view,
			geometryState.v3_view,
			geometryState.normal_view,
			feature,
			geometryInfo.opacity,
			geometryInfo.background_depth,
			geometryInfo.background,
			imageState.final_Ts,
			backwardInput.feature,
			backwardInput.normal,
			backwardInput.depth,
			backwardInput.distortion,
			lossInput.dL_dout_feature,
			lossInput.dL_dout_depth,
			lossInput.dL_dout_normal,
			lossInput.dL_dout_distortion,
			dL_dv1_view_ptr,
			dL_dv2_view_ptr,
			dL_dv3_view_ptr,
			backwardOutput.dL_dfeature,
			backwardOutput.dL_dopacity,
			backwardOutput.dL_dv_norm);
	}
	CHECK_CUDA(debug);

	BACKWARD::preprocessCUDA<<<(P + 255) / 256, 256>>>(
		W, H, P, geometryInfo.D, geometryInfo.M, geometryInfo.use_shs, geometryInfo.use_vertex_color, rich_info,
		cameraInfo.viewmatrix,
		cameraInfo.projmatrix,
		cameraInfo.campos,
		geometryInfo.vertex,
		geometryInfo.shs,
		backwardInput.radii,
		geometryState.clamped,
		dL_dv1_view_ptr,
		dL_dv2_view_ptr,
		dL_dv3_view_ptr,
		backwardOutput.dL_dfeature,
		backwardOutput.dL_dvertex,
		backwardOutput.dL_dshs);
	CHECK_CUDA(debug);
}