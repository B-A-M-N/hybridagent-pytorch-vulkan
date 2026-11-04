"""Dump the per-thread shared-memory view of the tiled matmul."""
import struct
import sys

import torch

sys.path.insert(0, "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan")
import torch_vulkan_ext as tvk  # noqa: E402


def main():
    tvk.init(0)
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    w = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    ab = tvk.VkBuffer(a.numel() * 4, "device"); ab.upload(a)
    wb = tvk.VkBuffer(w.numel() * 4, "device"); wb.upload(w)
    dbg = tvk.VkBuffer(4 * 6 * 4, "host")
    push = struct.pack("<III", 2, 2, 2)
    tvk.dispatch("mm_diag", [ab, wb, dbg], push, 1, 1, 1)
    v = dbg.download((4, 6))

    names = ["sA[ly][lx]", "sW[ly][lx]", "sA[ly][0]", "sA[ly][1]",
             "sW[lx][0]", "sW[lx][1]"]
    for tid in range(4):
        ly, lx = divmod(tid, 2)
        print(f"thread tid={tid} (ly={ly}, lx={lx}):")
        for i, n in enumerate(names):
            print(f"    {n:12s} = {v[tid, i].item():.1f}")

    print("\nexpected C = A@W.T = [[17,23],[39,53]]")
    print("  C[0,0] (ly=0,lx=0) = sA[0][0]*sW[0][0] + sA[0][1]*sW[0][1]"
          " = 1*5 + 2*6 = 17")
    print("  C[0,1] (ly=0,lx=1) = sA[0][0]*sW[1][0] + sA[0][1]*sW[1][1]"
          " = 1*7 + 2*8 = 23")


if __name__ == "__main__":
    main()
