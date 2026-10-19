# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import torch
import torch.cuda
from numba import jit, prange
from torch.autograd import Function
from numba import cuda
import math

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    """
    :param seq_len: The length of the sequence (both inputs are assumed to be of the same size)
    :param n_passes: 2 * seq_len - 1 (The number of anti-diagonals)
    """
    # # Each block processes one pair of examples
    # b = cuda.blockIdx.x
    # # We have as many threads as seq_len, because the most number of threads we need
    # # is equal to the number of elements on the largest anti-diagonal
    # tid = cuda.threadIdx.x

    # # Compute I, J, the indices from [0, seq_len)

    # # The row index is always the same as tid
    # I = tid

    global_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    b = global_idx // max_i    # batch 维度
    I = global_idx % max_i     # 当前行索引
    if b >= D.shape[0] or I >= max_i:
        return

    inv_gamma = 1.0 / gamma

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_passes):

        # The index is actually 'p - tid' but need to force it in-bounds
        J = max(0, min(p - I, max_j - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == p and (I < max_i and J < max_j):
            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                r0 = -R[b, i - 1, j - 1] * inv_gamma
                r1 = -R[b, i - 1, j] * inv_gamma
                r2 = -R[b, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
    # k = cuda.blockIdx.x
    # tid = cuda.threadIdx.x

    # # Indexing logic is the same as above, however, the anti-diagonal needs to
    # # progress backwards
    # I = tid
    global_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    k = global_idx // max_i
    I = global_idx % max_i
    if k >= D.shape[0] or I >= max_i:
        return

    for p in range(n_passes):
        # Reverse the order to make the loop go backward
        rev_p = n_passes - p - 1

        # convert tid to I, J, then i, j
        J = max(0, min(rev_p - I, max_j - 1))

        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == rev_p and (I < max_i and J < max_j):

            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf

            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) * inv_gamma)
                b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) * inv_gamma)
                c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTWPyTorch(Function):
    """
    Pure PyTorch anti-diagonal soft-DTW. No Numba CUDA JIT dependency.
    Works on any GPU architecture (Blackwell sm_120, Hopper sm_90, …).
    Algorithm is identical to the Numba kernel above.
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        B, N, M = D.shape
        dev, dtype = D.device, D.dtype
        inv_g = 1.0 / gamma

        R = torch.full((B, N + 2, M + 2), float('inf'), device=dev, dtype=dtype)
        R[:, 0, 0] = 0.0

        for p in range(N + M - 1):
            i_lo = max(1, p + 2 - M)
            i_hi = min(N, p + 1)
            ii = torch.arange(i_lo, i_hi + 1, device=dev)
            jj = (p + 2) - ii

            if bandwidth > 0:
                keep = (ii - jj).abs() <= bandwidth
                ii, jj = ii[keep], jj[keep]
            if ii.numel() == 0:
                continue

            r0 = R[:, ii - 1, jj - 1]   # (B, L)
            r1 = R[:, ii - 1, jj    ]
            r2 = R[:, ii,     jj - 1]

            v0, v1, v2 = -r0 * inv_g, -r1 * inv_g, -r2 * inv_g
            m = torch.maximum(torch.maximum(v0, v1), v2)

            # Guard: when all predecessors are +inf, m == -inf → exp(-inf-(-inf)) = NaN
            lse = torch.exp(v0 - m) + torch.exp(v1 - m) + torch.exp(v2 - m)
            softmin = torch.where(
                m.isneginf(),
                torch.full_like(m, float('inf')),
                -gamma * (torch.log(lse.clamp(min=1e-38)) + m),
            )

            R[:, ii, jj] = D[:, ii - 1, jj - 1] + softmin

        ctx.save_for_backward(
            D, R.clone(),
            torch.tensor(gamma,     device=dev, dtype=dtype),
            torch.tensor(bandwidth, device=dev, dtype=dtype),
        )
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        D, R, gamma_t, bw_t = ctx.saved_tensors
        gamma = gamma_t.item()
        bw    = bw_t.item()
        inv_g = 1.0 / gamma
        B, N, M = D.shape
        dev, dtype = D.device, D.dtype

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        R = R.clone()
        R[:, :, -1] = -float('inf')
        R[:, -1, :] = -float('inf')
        R[:, -1, -1] = R[:, -2, -2]

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1.0

        for p in range(N + M - 1):
            rev_p = (N + M - 2) - p
            i_lo  = max(1, rev_p + 2 - M)
            i_hi  = min(N, rev_p + 1)
            ii = torch.arange(i_lo, i_hi + 1, device=dev)
            jj = (rev_p + 2) - ii

            if bw > 0:
                keep   = (ii - jj).abs() <= bw
                ii, jj = ii[keep], jj[keep]
            if ii.numel() == 0:
                continue

            def _clamp_inf(t):
                return torch.where(t.isinf() & (t > 0), torch.full_like(t, -float('inf')), t)

            r_ij  = _clamp_inf(R[:, ii,     jj    ])
            r_i1j = _clamp_inf(R[:, ii + 1, jj    ])
            r_ij1 = _clamp_inf(R[:, ii,     jj + 1])
            r_i1j1= _clamp_inf(R[:, ii + 1, jj + 1])

            a = torch.exp((r_i1j  - r_ij - D_[:, ii + 1, jj    ]) * inv_g)
            b = torch.exp((r_ij1  - r_ij - D_[:, ii,     jj + 1]) * inv_g)
            c = torch.exp((r_i1j1 - r_ij - D_[:, ii + 1, jj + 1]) * inv_g)

            E[:, ii, jj] = (
                E[:, ii + 1, jj    ] * a +
                E[:, ii,     jj + 1] * b +
                E[:, ii + 1, jj + 1] * c
            )

        E = E[:, 1:N + 1, 1:M + 1]
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


class _SoftDTWCUDA(Function):
    """
    Original Numba CUDA implementation (kept for reference, no longer used).
    Numba 0.64.0 NVVM only supports up to sm_90; Blackwell (sm_120) is not supported.
    Use _SoftDTWPyTorch instead.
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = min(max(N, M), 512)
        blocks_per_grid = B * math.ceil(max(N, M) / threads_per_block)
        n_passes = 2 * max(N, M) - 1

        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        compute_softdtw_cuda[blocks_per_grid, threads_per_block](
                                                                    cuda.as_cuda_array(D.detach()),
                                                                    gamma.item(), bandwidth.item(), N, M, n_passes,
                                                                    cuda.as_cuda_array(R)
                                                                )
        ctx.save_for_backward(D, R.clone(), gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = min(max(N, M), 512)
        blocks_per_grid = B * math.ceil(max(N, M) / threads_per_block)
        n_passes = 2 * max(N, M) - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        R[:, :, -1] = -math.inf
        R[:, -1, :] = -math.inf
        R[:, -1, -1] = R[:, -2, -2]

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1

        compute_softdtw_backward_cuda[blocks_per_grid, threads_per_block](
                                                                            cuda.as_cuda_array(D_),
                                                                            cuda.as_cuda_array(R),
                                                                            1.0 / gamma.item(), bandwidth.item(), N, M, n_passes,
                                                                            cuda.as_cuda_array(E)
                                                                        )

        E = E[:, 1:N + 1, 1:M + 1]
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


# ----------------------------------------------------------------------------------------------------------------------
#
# The following is the CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
# Credit goes to Kanru Hua.
# I've added support for batching and pruning.
#
# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw(D, gamma, bandwidth):

    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for b in prange(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = - gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    # print("-------------------RRRRRRRRR------------------------")
    # print(R.shape)
    # print(R)
    # print("-------------------RRRRRRRRR------------------------")
    return R

# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw_backward(D_, R, gamma, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in prange(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):

                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    
    # print("-------------------EEEEEEEEEEE------------------------")
    # print(E[:, 1:N + 1, 1:M + 1])
    # print(E[:, 1:N + 1, 1:M + 1].shape)
    # print("-------------------EEEEEEEEEEE------------------------")
    return E[:, 1:N + 1, 1:M + 1]

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTW(Function):
    """
    CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        R = torch.Tensor(compute_softdtw(D_, g_, b_)).to(dev).type(dtype)
        # pdb.set_trace()
        ctx.save_for_backward(D, R, gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, g_, b_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None

# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """

    def __init__(self, use_cuda, gamma=1.0, contrastive_temp=1.0, normalize=False, bandwidth=None, dist_func=None):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param normalize: Flag indicating whether to perform normalization
                          (as discussed in https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790)
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        :param dist_func: Optional point-wise distance function to use. If 'None', then a default Euclidean distance function will be used.
        """
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.contrastive_temp = contrastive_temp
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

        # Set the distance function
        if dist_func is not None:
            self.dist_func = dist_func
        else:
            self.dist_func = SoftDTW._contrastive_dist_func

    def _get_func_dtw(self, x, y):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        # _SoftDTWPyTorch: pure PyTorch anti-diagonal DP, no Numba CUDA JIT.
        # Works on any GPU (Blackwell sm_120, Hopper sm_90, …) with no sequence-length limit.
        if self.use_cuda and x.is_cuda:
            return _SoftDTWPyTorch.apply
        return _SoftDTW.apply

    @staticmethod
    def _contrastive_dist_func(x, y, contrastive_temp):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """

        # n = x.size(1)
        # m = y.size(1)
        # d = x.size(2)
        # x = x.unsqueeze(2).expand(-1, n, m, d)
        # y = y.unsqueeze(1).expand(-1, n, m, d)
        # return torch.pow(x - y, 2).sum(3)
    
        # logit = torch.nn.functional.normalize(x, dim=-1) @ torch.nn.functional.normalize(y, dim=-1).permute(0, 2, 1)
        logit = x @ y.permute(0, 2, 1)
        logit = - torch.nn.functional.log_softmax(logit / contrastive_temp, dim=-1)

        return logit


    def forward(self, X, Y):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """

        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            # Stack everything up and run
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y, self.contrastive_temp)
            out = func_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)
        else:
            D_xy = self.dist_func(X, Y, self.contrastive_temp)
            return func_dtw(D_xy, self.gamma, self.bandwidth)

# ----------------------------------------------------------------------------------------------------------------------
def timed_run(a, b, sdtw):
    """
    Runs a and b through sdtw, and times the forward and backward passes.
    Assumes that a requires gradients.
    :return: timing, forward result, backward result
    """
    from timeit import default_timer as timer

    # Forward pass
    start = timer()
    forward = sdtw(a, b)
    end = timer()
    t = end - start

    grad_outputs = torch.ones_like(forward)

    # Backward
    start = timer()
    grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
    end = timer()

    # Total time
    t += end - start

    return t, forward, grads

# ----------------------------------------------------------------------------------------------------------------------
def profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward):
    sdtw = SoftDTW(False, gamma=1.0, normalize=False)
    sdtw_cuda = SoftDTW(True, gamma=1.0, normalize=False)
    n_iters = 6

    print("Profiling forward() + backward() times for batch_size={}, seq_len_a={}, seq_len_b={}, dims={}...".format(batch_size, seq_len_a, seq_len_b, dims))

    times_cpu = []
    times_gpu = []

    for i in range(n_iters):
        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        # GPU
        t_gpu, forward_gpu, backward_gpu = timed_run(a_gpu, b_gpu, sdtw_cuda)

        # CPU
        t_cpu, forward_cpu, backward_cpu = timed_run(a_cpu, b_cpu, sdtw)

        # Verify the results
        assert torch.allclose(forward_cpu, forward_gpu.cpu())
        assert torch.allclose(backward_cpu, backward_gpu.cpu(), atol=tol_backward)

        if i > 0:  # Ignore the first time we run, in case this is a cold start (because timings are off at a cold start of the script)
            times_cpu += [t_cpu]
            times_gpu += [t_gpu]

    # Average and log
    avg_cpu = np.mean(times_cpu)
    avg_gpu = np.mean(times_gpu)
    print("  CPU:     ", avg_cpu)
    print("  GPU:     ", avg_gpu)
    print("  Speedup: ", avg_cpu / avg_gpu)
    print()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from timeit import default_timer as timer

    torch.manual_seed(1234)

    # profile(128, 17, 15, 2, tol_backward=1e-6)
    # profile(512, 64, 64, 2, tol_backward=1e-4)
    # profile(512, 256, 256, 2, tol_backward=1e-3)
    speech_feat = torch.rand(1, 6, 4096)
    asr_emb = torch.rand(1, 3, 4096)
    module = SoftDTW(False)
    dis = ((speech_feat[:,-3:, :] - asr_emb) ** 2).sum()
    loss = module(speech_feat, asr_emb)
    # pdb.set_trace()