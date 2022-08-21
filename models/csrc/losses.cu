#include "utils.h"
#include <thrust/scan.h>


// for details of the formulae, please see https://arxiv.org/pdf/2206.05085.pdf

template <typename scalar_t>
__global__ void prefix_sums_kernel(
    const scalar_t* ws,
    const scalar_t* wts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    scalar_t* ws_prefix_sum,
    scalar_t* wts_prefix_sum
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // compute prefix sum of ws and ws*ts
    // [a0, a1, a2, a3, ...] -> [0, a0, a0+a1, a0+a1+a2, ...]
    thrust::exclusive_scan(thrust::device,
                           ws+start_idx,
                           ws+start_idx+N_samples,
                           ws_prefix_sum+start_idx);
    thrust::exclusive_scan(thrust::device,
                           wts+start_idx,
                           wts+start_idx+N_samples,
                           wts_prefix_sum+start_idx);
}


template <typename scalar_t>
__global__ void reduce_distortion_loss_kernel(
    const scalar_t* _loss,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> loss
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    loss[ray_idx] = thrust::reduce(thrust::device, 
                                   _loss+start_idx,
                                   _loss+start_idx+N_samples,
                                   (scalar_t)0);
}


std::vector<torch::Tensor> distortion_loss_fw_cu(
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
){
    const int N_rays = rays_a.size(0), N = ws.size(0);

    auto wts = ws * ts;

    auto ws_prefix_sum = torch::zeros({N}, ws.options());
    auto wts_prefix_sum = torch::zeros({N}, ws.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(ws.type(), "distortion_loss_fw_cu_prefix_sum", 
    ([&] {
        prefix_sums_kernel<scalar_t><<<blocks, threads>>>(
            ws.data_ptr<scalar_t>(),
            wts.data_ptr<scalar_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            ws_prefix_sum.data_ptr<scalar_t>(),
            wts_prefix_sum.data_ptr<scalar_t>()
        );
    }));

    auto _loss = 2*ws*(ts*ws_prefix_sum-wts_prefix_sum); // (N)
    _loss += 1.0f/3*ws*ws*deltas; // add the second term

    auto loss = torch::zeros({N_rays}, ws.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(ws.type(), "distortion_loss_fw_cu_reduce_loss", 
    ([&] {
        reduce_distortion_loss_kernel<scalar_t><<<blocks, threads>>>(
            _loss.data_ptr<scalar_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            loss.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {loss, ws_prefix_sum, wts_prefix_sum};
}


template <typename scalar_t>
__global__ void distortion_loss_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dloss,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ws_prefix_sum,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> wts_prefix_sum,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ws,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dws
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];
    const int end_idx = start_idx+N_samples-1;

    const scalar_t ws_total = ws_prefix_sum[end_idx]+ws[end_idx];
    const scalar_t wts_total = wts_prefix_sum[end_idx]+ws[end_idx]*ts[end_idx];
    // fill in dL_dws from start_idx to start_idx+N_samples
    for (int s=start_idx; s<=end_idx; s++){
        dL_dws[s] = dL_dloss[ray_idx] * 2 * (
            ts[s]*ws_prefix_sum[s] - wts_prefix_sum[s] +
            (s==end_idx?(scalar_t)0:
                        (wts_total-wts_prefix_sum[s+1]-ts[s]*(ws_total-ws_prefix_sum[s+1]))
            )
        );
        dL_dws[s] += dL_dloss[ray_idx] * (scalar_t)2/3*ws[s]*deltas[s];
    }
}


torch::Tensor distortion_loss_bw_cu(
    const torch::Tensor dL_dloss,
    const torch::Tensor ws_prefix_sum,
    const torch::Tensor wts_prefix_sum,
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
){
    const int N_rays = rays_a.size(0), N = ws.size(0);

    auto dL_dws = torch::zeros({N}, dL_dloss.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(ws.type(), "distortion_loss_bw_cu", 
    ([&] {
        distortion_loss_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dloss.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ws_prefix_sum.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            wts_prefix_sum.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            dL_dws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return dL_dws;
}