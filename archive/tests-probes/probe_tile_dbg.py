"""Dump the tiled kernel's per-tile indices to see what it computes."""
import struct
import sys

import torch

sys.path.insert(0, "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan")
import torch_vulkan_ext as tvk  # noqa: E402


def main():
    tvk.init(0)
    M, K, N = 4, 4, 4
    tiles_x = (N + 15) // 16
    tiles_y = (M + 15) // 16
    print(f"tiles_x={tiles_x} tiles_y={tiles_y} grid={tiles_x*tiles_y}")

    # identity-ish: a = I, w = I -> c should be I
    a = torch.zeros(M, K)
    for i in range(min(M, K)):
        a[i, i] = 1.0
    w = torch.zeros(N, K)
    for i in range(min(N, K)):
        w[i, i] = 1.0
    b = torch.zeros(N)

    ab = tvk.VkBuffer(a.numel() * 4, "device"); ab.upload(a)
    wb = tvk.VkBuffer(w.numel() * 4, "device"); wb.upload(w)
    bb = tvk.VkBuffer(b.numel() * 4, "device"); bb.upload(b)
    ob = tvk.VkBuffer(M * N * 4, "host")
    tvk.matmul_t_bias(ab, wb, bb, ob, M, K, N)
    print("identity test got:\n", ob.download((M, N)).tolist())
    print("expect:\n", (torch.eye(M, N)).tolist())


if __name__ == "__main__":
    main()
