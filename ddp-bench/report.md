# DDP Benchmark Report

- **Models**: 9
- **Modes**: 9
- **GPU speed ratio**: 2.65x (solo-0 / solo-1)

## Per-Model Results

### autoencoder

| Mode | Loss | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | Idle (s) |
|------|------|-----------|-------|--------------|------|------|----------|
| cpu-async | 0.838987 | 39.0 | 42 | 42.9 | 98% | 68% | 11.4 |
| cpu-cadence | 0.788722 | 40.0 | 41 | 44.3 | 96% | 71% | 12.3 |
| cpu-sync | 0.635434 | 104.6 | 2419 | 33.8 | 98% | 100% | 1.3 |
| nccl-async | 0.774669 | 41.0 | 48 | 0.0 | 97% | 60% | 15.1 |
| nccl-cadence | 0.757169 | 41.5 | 48 | 0.0 | 96% | 61% | 16.3 |
| nccl-sync | 0.570249 | 76.1 | 1347 | 0.0 | 55% | 99% | 33.8 |
| solo-0 | 0.378660 | 41.2 | 0 | 0.0 | 99% | 0% | 41.2 |
| solo-1 | 0.277702 | 144.1 | 0 | 0.0 | 0% | 100% | 144.1 |
| sync | 0.000000 | 46.2 | 5000 | 1.0 | 99% | 99% | 0.0 |

### convnet

| Mode | Loss | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | Idle (s) |
|------|------|-----------|-------|--------------|------|------|----------|
| cpu-async | 0.000003 | 55.8 | 15 | 60.5 | 99% | 60% | 21.6 |
| cpu-cadence | 0.000003 | 56.0 | 15 | 56.7 | 97% | 64% | 21.0 |
| cpu-sync | 0.000008 | 129.5 | 2495 | 42.9 | 100% | 100% | 0.0 |
| nccl-async | 0.000017 | 58.0 | 44 | 0.0 | 99% | 55% | 24.7 |
| nccl-cadence | 0.000005 | 57.5 | 42 | 0.0 | 99% | 56% | 24.5 |
| nccl-sync | 0.000011 | 95.7 | 1417 | 0.0 | 57% | 99% | 40.0 |
| solo-0 | 0.000003 | 64.1 | 0 | 0.0 | 99% | 0% | 64.9 |
| solo-1 | 0.000003 | 184.6 | 0 | 0.0 | 0% | 100% | 184.6 |
| sync | 0.000000 | 82.7 | 5000 | 2.5 | 99% | 99% | 0.6 |

### feedback

| Mode | Loss | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | Idle (s) |
|------|------|-----------|-------|--------------|------|------|----------|
| cpu-async | 0.970347 | 60.1 | 46 | 84.3 | 90% | 93% | 9.9 |
| cpu-cadence | 0.996130 | 58.7 | 43 | 80.9 | 93% | 87% | 10.2 |
| cpu-sync | 0.981649 | 208.9 | 2403 | 70.9 | 98% | 100% | 3.5 |
| nccl-async | 1.006450 | 61.0 | 81 | 0.0 | 92% | 79% | 15.1 |
| nccl-cadence | 0.996291 | 61.8 | 82 | 0.0 | 92% | 81% | 14.9 |
| nccl-sync | 1.010536 | 115.2 | 1747 | 0.0 | 73% | 99% | 30.9 |
| solo-0 | 1.105216 | 64.1 | 0 | 0.0 | 100% | 0% | 64.1 |
| solo-1 | 1.067329 | 210.7 | 0 | 0.0 | 0% | 100% | 210.7 |
| sync | 0.000000 | 178.9 | 5000 | 0.8 | 100% | 100% | 0.0 |

### linear

| Mode | Loss | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | Idle (s) |
|------|------|-----------|-------|--------------|------|------|----------|
| cpu-async | 0.069559 | 10.0 | 8 | 29.7 | 98% | 64% | 3.4 |
| cpu-cadence | 0.073857 | 9.7 | 8 | 28.1 | 98% | 69% | 2.8 |
| cpu-sync | 0.000934 | 63.7 | 2476 | 21.7 | 100% | 100% | 0.0 |
| nccl-async | 0.050067 | 11.4 | 60 | 0.0 | 96% | 51% | 5.1 |
| nccl-cadence | 0.044165 | 11.2 | 60 | 0.0 | 94% | 54% | 4.8 |
| nccl-sync | 0.000000 | 9.6 | 991 | 0.0 | 93% | 93% | 0.0 |
| solo-0 | 0.004251 | 3.2 | 0 | 0.0 | 79% | 4% | 3.6 |
| solo-1 | 0.003911 | 5.5 | 0 | 0.0 | 0% | 98% | 5.5 |
| sync | 0.000000 | 16.3 | 2000 | 0.4 | 97% | 97% | 2.0 |

### lstm

| Mode | Loss | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | Idle (s) |
|------|------|-----------|-------|--------------|------|------|----------|
| cpu-async | 0.000162 | 88.5 | 58 | 66.1 | 100% | 86% | 10.8 |
| cpu-cadence | 0.000138 | 92.7 | 86 | 63.3 | 100% | 87% | 10.4 |
| cpu-sync | 0.000285 | 138.4 | 2235 | 39.5 | 97% | 100% | 3.6 |
| nccl-async | 0.000041 | 89.4 | 116 | 0.0 | 99% | 84% | 11.4 |
| nccl-cadence | 0.000111 | 93.0 | 161 | 0.0 | 99% | 87% | 9.9 |
| nccl-sync | 0.000100 | 98.8 | 2190 | 0.0 | 94% | 99% | 4.5 |
| solo-0 | 0.000582 | 86.0 | 0 | 0.0 | 100% | 0% | 86.0 |
| solo-1 | 0.000609 | 97.3 | 0 | 0.0 | 0% | 100% | 97.3 |
| sync | 0.000000 | 188.9 | 5000 | 1.9 | 100% | 100% | 0.5 |

### mlp

| Mode | Loss | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | Idle (s) |
|------|------|-----------|-------|--------------|------|------|----------|
| cpu-async | 0.340855 | 217.2 | 13 | 1549.0 | 82% | 99% | 36.1 |
| cpu-cadence | 0.461151 | 210.3 | 13 | 1586.9 | 86% | 99% | 27.1 |
| cpu-sync | 0.026302 | 2682.1 | 1785 | 1318.8 | 46% | 55% | 2046.5 |
| nccl-async | 0.016438 | 310.9 | 110 | 0.0 | 55% | 99% | 135.1 |
| nccl-cadence | 0.016664 | 313.9 | 99 | 0.0 | 51% | 99% | 148.1 |
| nccl-sync | 0.006679 | 710.1 | 2493 | 0.0 | 100% | 100% | 3.5 |
| solo-0 | 0.002317 | 246.4 | 0 | 0.0 | 100% | 0% | 246.4 |
| solo-1 | 0.002317 | 670.1 | 0 | 0.0 | 1% | 100% | 663.9 |
| sync | 0.000000 | 1438.9 | 5000 | 3.0 | 100% | 100% | 0.7 |

### moe

| Mode | Loss | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | Idle (s) |
|------|------|-----------|-------|--------------|------|------|----------|
| cpu-async | 0.012112 | 66.3 | 12 | 424.4 | 90% | 99% | 6.8 |
| cpu-cadence | 0.003862 | 65.9 | 12 | 449.4 | 93% | 97% | 5.6 |
| cpu-sync | 0.001958 | 594.1 | 1060 | 441.3 | 82% | 90% | 50.5 |
| nccl-async | 0.002077 | 75.9 | 139 | 0.0 | 91% | 98% | 6.8 |
| nccl-cadence | 0.035825 | 77.1 | 134 | 0.0 | 90% | 96% | 8.8 |
| nccl-sync | 0.005086 | 260.2 | 2461 | 0.0 | 99% | 100% | 0.0 |
| solo-0 | 0.002669 | 76.7 | 0 | 0.0 | 100% | 0% | 76.7 |
| solo-1 | 0.002639 | 203.1 | 0 | 0.0 | 0% | 100% | 203.1 |
| sync | 0.000000 | 560.4 | 5000 | 7.9 | 100% | 100% | 0.0 |

### residual

| Mode | Loss | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | Idle (s) |
|------|------|-----------|-------|--------------|------|------|----------|
| cpu-async | 0.000724 | 89.6 | 7 | 728.7 | 93% | 99% | 5.0 |
| cpu-cadence | 0.001769 | 91.9 | 9 | 621.5 | 92% | 97% | 9.0 |
| cpu-sync | 0.025216 | 759.3 | 932 | 622.9 | 72% | 83% | 84.7 |
| nccl-async | 0.026851 | 102.3 | 135 | 0.0 | 91% | 97% | 10.9 |
| nccl-cadence | 0.027140 | 107.1 | 129 | 0.0 | 88% | 96% | 16.5 |
| nccl-sync | 0.012862 | 373.2 | 2484 | 0.0 | 100% | 100% | 0.8 |
| solo-0 | 3.968968 | 114.1 | 0 | 0.0 | 100% | 0% | 114.1 |
| solo-1 | 4.362538 | 285.3 | 0 | 0.0 | 0% | 100% | 285.3 |
| sync | 0.000000 | 816.8 | 5000 | 7.9 | 100% | 100% | 0.5 |

### transformer

| Mode | Loss | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | Idle (s) |
|------|------|-----------|-------|--------------|------|------|----------|
| cpu-async | 0.051662 | 46.6 | 65 | 41.0 | 99% | 88% | 5.0 |
| cpu-cadence | 0.051471 | 41.7 | 68 | 37.6 | 96% | 98% | 1.7 |
| cpu-sync | 0.054757 | 98.7 | 2365 | 29.8 | 98% | 100% | 1.4 |
| nccl-async | 0.090662 | 47.9 | 171 | 0.0 | 92% | 88% | 8.2 |
| nccl-cadence | 0.051109 | 42.1 | 158 | 0.0 | 99% | 92% | 2.5 |
| nccl-sync | 0.054537 | 56.9 | 2470 | 0.0 | 98% | 99% | 0.6 |
| solo-0 | 0.048915 | 66.4 | 0 | 0.0 | 100% | 0% | 66.4 |
| solo-1 | 0.047912 | 103.9 | 0 | 0.0 | 0% | 100% | 103.9 |
| sync | 0.000000 | 109.3 | 5000 | 5.0 | 100% | 100% | 0.6 |

## Speedup vs Sync

| Model | cpu-async | cpu-cadence | cpu-sync | nccl-async | nccl-cadence | nccl-sync | solo-0 | solo-1 |
|-------|-----------|-------------|----------|------------|--------------|-----------|--------|--------|
| autoencoder | 1.2x | 1.2x | 0.4x | 1.1x | 1.1x | 0.6x | 1.1x | 0.3x |
| convnet | 1.5x | 1.5x | 0.6x | 1.4x | 1.4x | 0.9x | 1.3x | 0.4x |
| feedback | 3.0x | 3.0x | 0.9x | 2.9x | 2.9x | 1.6x | 2.8x | 0.8x |
| linear | 1.6x | 1.7x | 0.3x | 1.4x | 1.5x | 1.7x | 5.1x | 3.0x |
| lstm | 2.1x | 2.0x | 1.4x | 2.1x | 2.0x | 1.9x | 2.2x | 1.9x |
| mlp | 6.6x | 6.8x | 0.5x | 4.6x | 4.6x | 2.0x | 5.8x | 2.1x |
| moe | 8.5x | 8.5x | 0.9x | 7.4x | 7.3x | 2.2x | 7.3x | 2.8x |
| residual | 9.1x | 8.9x | 1.1x | 8.0x | 7.6x | 2.2x | 7.2x | 2.9x |
| transformer | 2.3x | 2.6x | 1.1x | 2.3x | 2.6x | 1.9x | 1.6x | 1.1x |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### autoencoder

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 35.5 | 3.5 | epoch-boundary(4) |
| cpu-async | gpu1 | 28.3 | 2.9 | epoch-boundary(3) |
| cpu-async | gpu1 | 13.2 | 2.5 | epoch-boundary(1) |
| cpu-async | gpu1 | 20.8 | 2.5 | epoch-boundary(2) |
| cpu-cadence | gpu1 | 36.5 | 3.5 | epoch-boundary(4) |
| cpu-cadence | gpu1 | 29.3 | 2.9 | epoch-boundary(3) |
| cpu-cadence | gpu1 | 21.8 | 2.5 | epoch-boundary(2) |
| cpu-cadence | gpu1 | 14.7 | 1.9 | epoch-boundary(1) |
| cpu-cadence | gpu0 | 7.4 | 1.5 | epoch-boundary(0) |
| cpu-sync | gpu0 | 83.3 | 0.6 | epoch-boundary(3) |
| cpu-sync | gpu0 | 20.5 | 0.6 | epoch-boundary(0) |
| nccl-async | gpu1 | 36.0 | 5.0 | epoch-boundary(4) |
| nccl-async | gpu1 | 20.8 | 3.6 | epoch-boundary(2) |
| nccl-async | gpu1 | 28.9 | 3.4 | epoch-boundary(3) |
| nccl-async | gpu1 | 13.1 | 3.2 | epoch-boundary(1) |
| nccl-cadence | gpu1 | 37.0 | 4.4 | epoch-boundary(4) |
| nccl-cadence | gpu1 | 28.6 | 4.3 | epoch-boundary(3) |
| nccl-cadence | gpu1 | 13.8 | 3.1 | epoch-boundary(1) |
| nccl-cadence | gpu1 | 21.8 | 2.9 | epoch-boundary(2) |
| nccl-cadence | gpu0 | 8.0 | 1.0 | epoch-boundary(0) |
| nccl-sync | gpu0 | 68.7 | 7.5 | epoch-boundary(4) |
| nccl-sync | gpu0 | 38.3 | 7.2 | epoch-boundary(2) |
| nccl-sync | gpu0 | 53.4 | 7.2 | epoch-boundary(3) |
| nccl-sync | gpu0 | 9.0 | 6.5 | epoch-boundary(0) |
| nccl-sync | gpu0 | 24.9 | 5.4 | epoch-boundary(1) |

### convnet

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 27.6 | 5.6 | epoch-boundary(2) |
| cpu-async | gpu1 | 39.1 | 5.6 | epoch-boundary(3) |
| cpu-async | gpu1 | 50.3 | 5.5 | epoch-boundary(4) |
| cpu-async | gpu1 | 16.8 | 4.9 | epoch-boundary(1) |
| cpu-cadence | gpu1 | 50.2 | 5.8 | epoch-boundary(4) |
| cpu-cadence | gpu1 | 39.6 | 5.2 | epoch-boundary(3) |
| cpu-cadence | gpu1 | 29.2 | 4.3 | epoch-boundary(2) |
| cpu-cadence | gpu1 | 18.1 | 4.3 | epoch-boundary(1) |
| cpu-cadence | gpu0 | 10.0 | 1.5 | epoch-boundary(0) |
| nccl-async | gpu1 | 38.9 | 6.9 | epoch-boundary(3) |
| nccl-async | gpu1 | 51.4 | 6.6 | epoch-boundary(4) |
| nccl-async | gpu1 | 16.5 | 5.9 | epoch-boundary(1) |
| nccl-async | gpu1 | 28.3 | 5.3 | epoch-boundary(2) |
| nccl-cadence | gpu1 | 38.2 | 7.2 | epoch-boundary(3) |
| nccl-cadence | gpu1 | 50.9 | 6.6 | epoch-boundary(4) |
| nccl-cadence | gpu1 | 16.8 | 5.4 | epoch-boundary(1) |
| nccl-cadence | gpu1 | 28.8 | 4.7 | epoch-boundary(2) |
| nccl-sync | gpu0 | 87.0 | 8.6 | epoch-boundary(4) |
| nccl-sync | gpu0 | 48.7 | 8.6 | epoch-boundary(2) |
| nccl-sync | gpu0 | 67.9 | 8.4 | epoch-boundary(3) |
| nccl-sync | gpu0 | 11.7 | 7.8 | epoch-boundary(0) |
| nccl-sync | gpu0 | 31.7 | 6.5 | epoch-boundary(1) |

### feedback

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu0 | 9.8 | 6.0 | epoch-boundary(0) |
| cpu-async | gpu1 | 46.5 | 3.1 | epoch-boundary(3) |
| cpu-async | gpu1 | 36.6 | 0.8 | epoch-boundary(2) |
| cpu-cadence | gpu0 | 9.9 | 4.0 | epoch-boundary(0) |
| cpu-cadence | gpu1 | 56.3 | 2.4 | epoch-boundary(4) |
| cpu-cadence | gpu1 | 44.7 | 2.4 | epoch-boundary(3) |
| cpu-cadence | gpu1 | 34.1 | 1.5 | epoch-boundary(2) |
| cpu-sync | gpu0 | 124.7 | 0.8 | epoch-boundary(2) |
| cpu-sync | gpu0 | 166.7 | 0.8 | epoch-boundary(3) |
| cpu-sync | gpu0 | 41.1 | 0.7 | epoch-boundary(0) |
| cpu-sync | gpu0 | 208.2 | 0.6 | epoch-boundary(4) |
| cpu-sync | gpu0 | 83.2 | 0.5 | epoch-boundary(1) |
| nccl-async | gpu0 | 10.6 | 4.6 | epoch-boundary(0) |
| nccl-async | gpu1 | 57.6 | 3.4 | epoch-boundary(4) |
| nccl-async | gpu1 | 35.3 | 2.6 | epoch-boundary(2) |
| nccl-async | gpu1 | 24.1 | 2.4 | epoch-boundary(1) |
| nccl-async | gpu1 | 47.0 | 2.2 | epoch-boundary(3) |
| nccl-cadence | gpu0 | 10.8 | 4.7 | epoch-boundary(0) |
| nccl-cadence | gpu1 | 35.0 | 3.2 | epoch-boundary(2) |
| nccl-cadence | gpu1 | 47.1 | 3.0 | epoch-boundary(3) |
| nccl-cadence | gpu1 | 59.7 | 2.2 | epoch-boundary(4) |
| nccl-cadence | gpu1 | 24.9 | 1.4 | epoch-boundary(1) |
| nccl-sync | gpu0 | 39.4 | 6.6 | epoch-boundary(1) |
| nccl-sync | gpu0 | 85.7 | 6.2 | epoch-boundary(3) |
| nccl-sync | gpu0 | 108.6 | 6.2 | epoch-boundary(4) |
| nccl-sync | gpu0 | 62.6 | 6.1 | epoch-boundary(2) |
| nccl-sync | gpu0 | 17.5 | 5.8 | epoch-boundary(0) |

### linear

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 7.1 | 1.0 | epoch-boundary(3) |
| cpu-async | gpu1 | 9.0 | 0.9 | epoch-boundary(4) |
| cpu-async | gpu1 | 5.0 | 0.8 | epoch-boundary(2) |
| cpu-async | gpu1 | 3.2 | 0.6 | epoch-boundary(1) |
| cpu-cadence | gpu1 | 7.0 | 0.8 | epoch-boundary(3) |
| cpu-cadence | gpu1 | 5.1 | 0.7 | epoch-boundary(2) |
| cpu-cadence | gpu1 | 9.0 | 0.6 | epoch-boundary(4) |
| cpu-cadence | gpu1 | 3.2 | 0.6 | epoch-boundary(1) |
| nccl-async | gpu1 | 9.9 | 1.5 | epoch-boundary(4) |
| nccl-async | gpu1 | 5.6 | 1.2 | epoch-boundary(2) |
| nccl-async | gpu1 | 7.7 | 1.2 | epoch-boundary(3) |
| nccl-async | gpu1 | 3.4 | 1.1 | epoch-boundary(1) |
| nccl-cadence | gpu1 | 9.8 | 1.4 | epoch-boundary(4) |
| nccl-cadence | gpu1 | 5.4 | 1.2 | epoch-boundary(2) |
| nccl-cadence | gpu1 | 7.6 | 1.1 | epoch-boundary(3) |
| nccl-cadence | gpu1 | 3.4 | 1.0 | epoch-boundary(1) |
| solo-0 | gpu1 | 0.6 | 2.6 | epoch-boundary(0) |

### lstm

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 50.3 | 3.4 | epoch-boundary(2) |
| cpu-async | gpu1 | 32.4 | 3.0 | epoch-boundary(1) |
| cpu-async | gpu1 | 86.2 | 2.3 | epoch-boundary(4) |
| cpu-async | gpu1 | 68.8 | 2.1 | epoch-boundary(3) |
| cpu-cadence | gpu1 | 68.6 | 6.5 | epoch-boundary(3) |
| cpu-cadence | gpu1 | 34.2 | 2.3 | epoch-boundary(1) |
| cpu-cadence | gpu1 | 53.0 | 1.6 | epoch-boundary(2) |
| cpu-sync | gpu0 | 82.4 | 1.2 | epoch-boundary(2) |
| cpu-sync | gpu0 | 26.7 | 1.0 | epoch-boundary(0) |
| cpu-sync | gpu0 | 137.6 | 0.8 | epoch-boundary(4) |
| cpu-sync | gpu0 | 110.8 | 0.6 | epoch-boundary(3) |
| nccl-async | gpu1 | 49.1 | 3.5 | epoch-boundary(2) |
| nccl-async | gpu1 | 32.7 | 3.1 | epoch-boundary(1) |
| nccl-async | gpu1 | 67.3 | 2.8 | epoch-boundary(3) |
| nccl-async | gpu1 | 87.3 | 2.1 | epoch-boundary(4) |
| nccl-cadence | gpu1 | 53.0 | 3.2 | epoch-boundary(2) |
| nccl-cadence | gpu1 | 90.4 | 2.7 | epoch-boundary(4) |
| nccl-cadence | gpu1 | 34.4 | 2.2 | epoch-boundary(1) |
| nccl-cadence | gpu1 | 73.1 | 1.0 | epoch-boundary(3) |
| nccl-cadence | gpu1 | 17.7 | 0.8 | epoch-boundary(0) |
| nccl-sync | gpu0 | 58.4 | 1.5 | epoch-boundary(2) |
| nccl-sync | gpu0 | 77.6 | 1.4 | epoch-boundary(3) |
| nccl-sync | gpu0 | 19.3 | 0.9 | epoch-boundary(0) |
| nccl-sync | gpu0 | 98.0 | 0.8 | epoch-boundary(4) |

### mlp

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu0 | 124.0 | 8.7 | epoch-boundary(2) |
| cpu-async | gpu0 | 79.7 | 8.6 | epoch-boundary(1) |
| cpu-async | gpu0 | 37.8 | 6.7 | epoch-boundary(0) |
| cpu-async | gpu0 | 168.4 | 5.9 | epoch-boundary(3) |
| cpu-async | gpu0 | 213.4 | 3.0 | epoch-boundary(4) |
| cpu-async | gpu0 | 210.0 | 1.3 | epoch-boundary(4) |
| cpu-async | gpu0 | 212.2 | 0.8 | cpu-avg |
| cpu-cadence | gpu0 | 37.4 | 7.3 | epoch-boundary(0) |
| cpu-cadence | gpu0 | 81.5 | 4.8 | epoch-boundary(1) |
| cpu-cadence | gpu0 | 163.1 | 4.7 | epoch-boundary(3) |
| cpu-cadence | gpu0 | 122.4 | 4.5 | epoch-boundary(2) |
| cpu-cadence | gpu0 | 204.0 | 2.2 | epoch-boundary(4) |
| cpu-cadence | gpu0 | 207.9 | 1.5 | epoch-boundary(4) |
| cpu-cadence | gpu0 | 206.8 | 1.0 | cpu-avg |
| cpu-sync | gpu0 | 2661.2 | 19.9 | epoch-boundary(4) |
| cpu-sync | gpu0 | 2143.6 | 19.5 | epoch-boundary(3) |
| cpu-sync | gpu0 | 1619.3 | 19.3 | epoch-boundary(2) |
| cpu-sync | gpu0 | 1094.1 | 18.8 | epoch-boundary(1) |
| cpu-sync | gpu0 | 562.7 | 17.5 | epoch-boundary(0) |
| cpu-sync | gpu0 | 537.9 | 1.2 | cpu-avg |
| cpu-sync | gpu0 | 23.8 | 1.1 | cpu-avg |
| cpu-sync | gpu0 | 220.3 | 1.1 | cpu-avg |
| cpu-sync | gpu0 | 274.6 | 1.0 | cpu-avg |
| cpu-sync | gpu0 | 295.6 | 1.0 | cpu-avg |
| nccl-async | gpu0 | 269.6 | 39.7 | epoch-boundary(4) |
| nccl-async | gpu0 | 204.4 | 35.0 | epoch-boundary(3) |
| nccl-async | gpu0 | 138.4 | 34.4 | epoch-boundary(2) |
| nccl-async | gpu0 | 36.5 | 10.6 | epoch-boundary(0) |
| nccl-async | gpu0 | 47.5 | 7.2 | unexplained |
| nccl-async | gpu0 | 57.1 | 3.8 | unexplained |
| nccl-async | gpu0 | 55.6 | 1.1 | unexplained |
| nccl-async | gpu0 | 309.9 | 1.0 | unexplained |
| nccl-async | gpu0 | 61.8 | 0.7 | epoch-boundary(0) |
| nccl-async | gpu0 | 268.6 | 0.5 | epoch-boundary(4) |
| nccl-cadence | gpu0 | 268.3 | 44.5 | epoch-boundary(4) |
| nccl-cadence | gpu0 | 203.7 | 35.9 | epoch-boundary(3) |
| nccl-cadence | gpu0 | 138.2 | 34.6 | epoch-boundary(2) |
| nccl-cadence | gpu0 | 34.1 | 30.4 | epoch-boundary(0) |
| nccl-cadence | gpu1 | 312.7 | 1.2 | unexplained |
| nccl-cadence | gpu0 | 137.2 | 0.7 | epoch-boundary(2) |
| nccl-sync | gpu0 | 708.9 | 1.2 | unexplained |
| nccl-sync | gpu1 | 708.3 | 0.6 | epoch-boundary(4) |
| solo-1 | gpu0 | 575.8 | 71.6 | unexplained |
| solo-1 | gpu0 | 371.1 | 68.3 | epoch-boundary(2) |
| solo-1 | gpu0 | 439.6 | 40.8 | unexplained |
| solo-1 | gpu0 | 527.9 | 28.5 | epoch-boundary(3) |
| solo-1 | gpu0 | 647.9 | 22.2 | epoch-boundary(4) |
| solo-1 | gpu0 | 501.3 | 16.4 | unexplained |
| solo-1 | gpu0 | 484.2 | 16.0 | unexplained |
| solo-1 | gpu0 | 353.8 | 15.3 | unexplained |
| solo-1 | gpu0 | 558.0 | 9.4 | unexplained |
| solo-1 | gpu0 | 518.5 | 9.3 | unexplained |

### moe

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu0 | 11.7 | 3.0 | epoch-boundary(0) |
| cpu-async | gpu0 | 51.9 | 1.2 | epoch-boundary(3) |
| cpu-async | gpu0 | 65.0 | 1.1 | epoch-boundary(4) |
| cpu-async | gpu0 | 39.1 | 0.8 | epoch-boundary(2) |
| cpu-async | gpu1 | 26.8 | 0.7 | epoch-boundary(1) |
| cpu-cadence | gpu0 | 12.0 | 3.6 | epoch-boundary(0) |
| cpu-cadence | gpu1 | 27.6 | 1.1 | epoch-boundary(1) |
| cpu-cadence | gpu0 | 64.9 | 0.8 | epoch-boundary(4) |
| cpu-sync | gpu0 | 346.3 | 10.8 | epoch-boundary(2) |
| cpu-sync | gpu0 | 108.5 | 10.0 | epoch-boundary(0) |
| cpu-sync | gpu0 | 230.1 | 9.9 | epoch-boundary(1) |
| cpu-sync | gpu0 | 463.1 | 8.3 | epoch-boundary(3) |
| cpu-sync | gpu0 | 589.7 | 4.2 | epoch-boundary(4) |
| cpu-sync | gpu0 | 584.5 | 3.7 | epoch-boundary(4) |
| cpu-sync | gpu0 | 460.3 | 2.5 | cpu-avg |
| cpu-sync | gpu0 | 588.5 | 1.0 | unexplained |
| nccl-async | gpu0 | 14.8 | 4.6 | epoch-boundary(0) |
| nccl-async | gpu0 | 33.0 | 1.6 | epoch-boundary(1) |
| nccl-async | gpu0 | 47.8 | 0.6 | epoch-boundary(2) |
| nccl-cadence | gpu0 | 13.4 | 5.8 | epoch-boundary(0) |
| nccl-cadence | gpu1 | 74.7 | 1.8 | epoch-boundary(4) |
| nccl-cadence | gpu0 | 32.4 | 1.1 | epoch-boundary(1) |

### residual

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu0 | 16.3 | 4.2 | epoch-boundary(0) |
| cpu-async | gpu0 | 88.4 | 0.8 | epoch-boundary(4) |
| cpu-cadence | gpu0 | 16.5 | 4.4 | epoch-boundary(0) |
| cpu-cadence | gpu0 | 36.5 | 2.6 | epoch-boundary(1) |
| cpu-cadence | gpu1 | 73.0 | 2.1 | epoch-boundary(3) |
| cpu-sync | gpu0 | 292.4 | 17.6 | epoch-boundary(1) |
| cpu-sync | gpu0 | 442.9 | 17.3 | epoch-boundary(2) |
| cpu-sync | gpu0 | 742.0 | 16.9 | epoch-boundary(4) |
| cpu-sync | gpu0 | 144.4 | 16.6 | epoch-boundary(0) |
| cpu-sync | gpu0 | 596.1 | 16.3 | epoch-boundary(3) |
| nccl-async | gpu0 | 18.2 | 7.8 | epoch-boundary(0) |
| nccl-async | gpu1 | 101.2 | 0.6 | epoch-boundary(4) |
| nccl-async | gpu1 | 63.7 | 0.6 | epoch-boundary(2) |
| nccl-async | gpu1 | 44.8 | 0.6 | epoch-boundary(1) |
| nccl-async | gpu0 | 101.8 | 0.5 | unexplained |
| nccl-cadence | gpu0 | 16.5 | 11.0 | epoch-boundary(0) |
| nccl-cadence | gpu1 | 46.4 | 2.2 | epoch-boundary(1) |
| nccl-cadence | gpu1 | 105.3 | 1.4 | epoch-boundary(4) |
| nccl-cadence | gpu0 | 86.6 | 0.8 | epoch-boundary(3) |
| nccl-cadence | gpu0 | 67.4 | 0.6 | epoch-boundary(2) |

### transformer

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 43.1 | 3.5 | epoch-boundary(4) |
| cpu-async | gpu1 | 25.2 | 1.5 | epoch-boundary(2) |
| cpu-cadence | gpu0 | 7.9 | 1.1 | epoch-boundary(0) |
| cpu-cadence | gpu1 | 33.1 | 0.5 | epoch-boundary(3) |
| cpu-sync | gpu0 | 19.1 | 0.8 | epoch-boundary(0) |
| cpu-sync | gpu0 | 79.0 | 0.6 | epoch-boundary(3) |
| nccl-async | gpu1 | 17.5 | 3.9 | epoch-boundary(1) |
| nccl-async | gpu0 | 28.3 | 3.1 | epoch-boundary(2) |
| nccl-async | gpu1 | 47.3 | 0.6 | epoch-boundary(4) |
| nccl-async | gpu1 | 38.9 | 0.6 | epoch-boundary(3) |
| nccl-cadence | gpu1 | 40.1 | 2.0 | epoch-boundary(4) |
| nccl-cadence | gpu1 | 16.7 | 0.5 | epoch-boundary(1) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| autoencoder | cpu-async | gpu1 | 11.4s | 0.0s | 0.0s | 0.0s | 11.4s |
| autoencoder | cpu-cadence | gpu0 | 1.5s | 0.0s | 0.0s | 0.0s | 1.5s |
| autoencoder | cpu-cadence | gpu1 | 10.7s | 0.0s | 0.0s | 0.0s | 10.7s |
| autoencoder | cpu-sync | gpu0 | 1.3s | 0.0s | 0.0s | 0.0s | 1.3s |
| autoencoder | nccl-async | gpu1 | 15.1s | 0.0s | 0.0s | 0.0s | 15.1s |
| autoencoder | nccl-cadence | gpu0 | 1.0s | 0.0s | 0.0s | 0.0s | 1.0s |
| autoencoder | nccl-cadence | gpu1 | 14.7s | 0.0s | 0.0s | 0.0s | 15.2s |
| autoencoder | nccl-sync | gpu0 | 33.8s | 0.0s | 0.0s | 0.0s | 33.8s |
| convnet | cpu-async | gpu1 | 21.6s | 0.0s | 0.0s | 0.0s | 21.6s |
| convnet | cpu-cadence | gpu0 | 1.5s | 0.0s | 0.0s | 0.0s | 1.5s |
| convnet | cpu-cadence | gpu1 | 19.6s | 0.0s | 0.0s | 0.0s | 19.6s |
| convnet | nccl-async | gpu1 | 24.7s | 0.0s | 0.0s | 0.0s | 24.7s |
| convnet | nccl-cadence | gpu0 | 0.0s | 0.0s | 0.0s | 0.0s | 0.6s |
| convnet | nccl-cadence | gpu1 | 24.0s | 0.0s | 0.0s | 0.0s | 24.0s |
| convnet | nccl-sync | gpu0 | 40.0s | 0.0s | 0.0s | 0.0s | 40.0s |
| convnet | sync | gpu1 | 0.0s | 0.0s | 0.0s | 0.0s | 0.6s |
| feedback | cpu-async | gpu0 | 6.0s | 0.0s | 0.0s | 0.0s | 6.0s |
| feedback | cpu-async | gpu1 | 3.9s | 0.0s | 0.0s | 0.0s | 3.9s |
| feedback | cpu-cadence | gpu0 | 4.0s | 0.0s | 0.0s | 0.0s | 4.0s |
| feedback | cpu-cadence | gpu1 | 6.2s | 0.0s | 0.0s | 0.0s | 6.2s |
| feedback | cpu-sync | gpu0 | 3.5s | 0.0s | 0.0s | 0.0s | 3.5s |
| feedback | nccl-async | gpu0 | 4.6s | 0.0s | 0.0s | 0.0s | 4.6s |
| feedback | nccl-async | gpu1 | 10.5s | 0.0s | 0.0s | 0.0s | 10.5s |
| feedback | nccl-cadence | gpu0 | 4.7s | 0.0s | 0.0s | 0.0s | 4.7s |
| feedback | nccl-cadence | gpu1 | 9.8s | 0.0s | 0.0s | 0.0s | 10.3s |
| feedback | nccl-sync | gpu0 | 30.9s | 0.0s | 0.0s | 0.0s | 30.9s |
| linear | cpu-async | gpu1 | 3.4s | 0.0s | 0.0s | 0.0s | 3.4s |
| linear | cpu-cadence | gpu1 | 2.8s | 0.0s | 0.0s | 0.0s | 2.8s |
| linear | nccl-async | gpu1 | 5.1s | 0.0s | 0.0s | 0.0s | 5.1s |
| linear | nccl-cadence | gpu1 | 4.8s | 0.0s | 0.0s | 0.0s | 4.8s |
| linear | sync | gpu0 | 0.0s | 0.0s | 0.0s | 0.0s | 0.8s |
| linear | sync | gpu1 | 0.0s | 0.0s | 0.0s | 0.0s | 1.1s |
| lstm | cpu-async | gpu1 | 10.8s | 0.0s | 0.0s | 0.0s | 10.8s |
| lstm | cpu-cadence | gpu1 | 10.4s | 0.0s | 0.0s | 0.0s | 10.4s |
| lstm | cpu-sync | gpu0 | 3.6s | 0.0s | 0.0s | 0.0s | 3.6s |
| lstm | nccl-async | gpu1 | 11.4s | 0.0s | 0.0s | 0.0s | 11.4s |
| lstm | nccl-cadence | gpu1 | 9.9s | 0.0s | 0.0s | 0.0s | 9.9s |
| lstm | nccl-sync | gpu0 | 4.5s | 0.0s | 0.0s | 0.0s | 4.5s |
| lstm | sync | gpu0 | 0.0s | 0.0s | 0.0s | 0.0s | 0.5s |
| mlp | cpu-async | gpu0 | 34.3s | 0.0s | 0.8s | 0.0s | 35.2s |
| mlp | cpu-async | gpu1 | 0.0s | 0.0s | 0.0s | 0.0s | 0.9s |
| mlp | cpu-cadence | gpu0 | 25.0s | 0.0s | 1.0s | 0.0s | 26.0s |
| mlp | cpu-cadence | gpu1 | 0.0s | 0.0s | 0.0s | 0.0s | 1.1s |
| mlp | cpu-sync | gpu0 | 97.9s | 0.0s | 1063.7s | 0.0s | 1161.6s |
| mlp | cpu-sync | gpu1 | 2.9s | 0.0s | 881.0s | 0.0s | 884.8s |
| mlp | nccl-async | gpu0 | 120.9s | 0.0s | 0.0s | 13.1s | 134.0s |
| mlp | nccl-async | gpu1 | 0.0s | 0.0s | 0.0s | 0.0s | 1.1s |
| mlp | nccl-cadence | gpu0 | 146.0s | 0.0s | 0.0s | 0.0s | 146.0s |
| mlp | nccl-cadence | gpu1 | 0.0s | 0.0s | 0.0s | 1.2s | 2.1s |
| mlp | nccl-sync | gpu0 | 0.0s | 0.0s | 0.0s | 1.2s | 1.2s |
| mlp | nccl-sync | gpu1 | 0.6s | 0.0s | 0.0s | 0.0s | 2.3s |
| mlp | sync | gpu1 | 0.0s | 0.0s | 0.0s | 0.0s | 0.7s |
| moe | cpu-async | gpu0 | 6.1s | 0.0s | 0.0s | 0.0s | 6.1s |
| moe | cpu-async | gpu1 | 0.7s | 0.0s | 0.0s | 0.0s | 0.7s |
| moe | cpu-cadence | gpu0 | 4.5s | 0.0s | 0.0s | 0.0s | 4.5s |
| moe | cpu-cadence | gpu1 | 1.1s | 0.0s | 0.0s | 0.0s | 1.1s |
| moe | cpu-sync | gpu0 | 47.0s | 0.0s | 2.5s | 1.0s | 50.5s |
| moe | nccl-async | gpu0 | 6.8s | 0.0s | 0.0s | 0.0s | 6.8s |
| moe | nccl-cadence | gpu0 | 7.0s | 0.0s | 0.0s | 0.0s | 7.0s |
| moe | nccl-cadence | gpu1 | 1.8s | 0.0s | 0.0s | 0.0s | 1.8s |
| residual | cpu-async | gpu0 | 5.0s | 0.0s | 0.0s | 0.0s | 5.0s |
| residual | cpu-cadence | gpu0 | 6.9s | 0.0s | 0.0s | 0.0s | 6.9s |
| residual | cpu-cadence | gpu1 | 2.1s | 0.0s | 0.0s | 0.0s | 2.1s |
| residual | cpu-sync | gpu0 | 84.7s | 0.0s | 0.0s | 0.0s | 84.7s |
| residual | nccl-async | gpu0 | 7.8s | 0.0s | 0.0s | 0.5s | 8.4s |
| residual | nccl-async | gpu1 | 1.9s | 0.0s | 0.0s | 0.0s | 2.5s |
| residual | nccl-cadence | gpu0 | 12.4s | 0.0s | 0.0s | 0.0s | 12.4s |
| residual | nccl-cadence | gpu1 | 3.6s | 0.0s | 0.0s | 0.0s | 4.2s |
| residual | nccl-sync | gpu1 | 0.0s | 0.0s | 0.0s | 0.0s | 0.8s |
| residual | sync | gpu1 | 0.0s | 0.0s | 0.0s | 0.0s | 0.5s |
| transformer | cpu-async | gpu1 | 5.0s | 0.0s | 0.0s | 0.0s | 5.0s |
| transformer | cpu-cadence | gpu0 | 1.1s | 0.0s | 0.0s | 0.0s | 1.1s |
| transformer | cpu-cadence | gpu1 | 0.5s | 0.0s | 0.0s | 0.0s | 0.5s |
| transformer | cpu-sync | gpu0 | 1.4s | 0.0s | 0.0s | 0.0s | 1.4s |
| transformer | nccl-async | gpu0 | 3.1s | 0.0s | 0.0s | 0.0s | 3.1s |
| transformer | nccl-async | gpu1 | 5.1s | 0.0s | 0.0s | 0.0s | 5.1s |
| transformer | nccl-cadence | gpu1 | 2.5s | 0.0s | 0.0s | 0.0s | 2.5s |
| transformer | nccl-sync | gpu0 | 0.0s | 0.0s | 0.0s | 0.0s | 0.6s |
| transformer | sync | gpu0 | 0.0s | 0.0s | 0.0s | 0.0s | 0.6s |

## ElChe Calibration

| Model | Mode | Anchor Changes | Throttles | Syncs | CPU Avgs | Avg CPU Avg (ms) |
|-------|------|---------------|-----------|-------|---------|----------------|
| autoencoder | cpu-async | 15 | 0 | 42 | 42 | 42.9 |
| autoencoder | cpu-cadence | 19 | 0 | 41 | 41 | 44.3 |
| autoencoder | cpu-sync | 1 | 0 | 2419 | 2419 | 33.8 |
| convnet | cpu-async | 14 | 0 | 15 | 15 | 60.5 |
| convnet | cpu-cadence | 15 | 0 | 15 | 15 | 56.7 |
| convnet | cpu-sync | 1 | 0 | 2495 | 2495 | 42.9 |
| feedback | cpu-async | 13 | 0 | 46 | 46 | 84.3 |
| feedback | cpu-cadence | 17 | 0 | 43 | 43 | 80.9 |
| feedback | cpu-sync | 2 | 0 | 2403 | 2403 | 70.9 |
| linear | cpu-async | 5 | 0 | 8 | 8 | 29.7 |
| linear | cpu-cadence | 6 | 0 | 8 | 8 | 28.1 |
| linear | cpu-sync | 1 | 0 | 2476 | 2476 | 21.7 |
| lstm | cpu-async | 32 | 0 | 58 | 58 | 66.1 |
| lstm | cpu-cadence | 29 | 0 | 86 | 86 | 63.3 |
| lstm | cpu-sync | 6 | 0 | 2235 | 2235 | 39.5 |
| mlp | cpu-async | 7 | 0 | 13 | 13 | 1549.0 |
| mlp | cpu-cadence | 7 | 0 | 13 | 13 | 1586.9 |
| mlp | cpu-sync | 3 | 0 | 1785 | 1785 | 1318.8 |
| moe | cpu-async | 10 | 0 | 12 | 12 | 424.4 |
| moe | cpu-cadence | 7 | 0 | 12 | 12 | 449.4 |
| moe | cpu-sync | 5 | 0 | 1060 | 1060 | 441.3 |
| residual | cpu-async | 2 | 0 | 7 | 7 | 728.7 |
| residual | cpu-cadence | 3 | 0 | 9 | 9 | 621.5 |
| residual | cpu-sync | 7 | 0 | 932 | 932 | 622.9 |
| transformer | cpu-async | 12 | 0 | 65 | 65 | 41.0 |
| transformer | cpu-cadence | 8 | 0 | 68 | 68 | 37.6 |
| transformer | cpu-sync | 2 | 0 | 2365 | 2365 | 29.8 |

