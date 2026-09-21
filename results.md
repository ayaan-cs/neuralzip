
| model | english.txt (517,537 B) | python.txt (293,103 B) | random.bin (32,768 B) |
|---|---:|---:|---:|
| gzip -9 | 2.251 bpb | 1.861 bpb | 8.004 bpb |
| bzip2 | 1.622 bpb (72% of gzip) | 1.569 bpb (84% of gzip) | 8.119 bpb (101% of gzip) |
| xz (lzma) | 1.579 bpb (70% of gzip) | 1.642 bpb (88% of gzip) | 8.015 bpb (100% of gzip) |
| order0 | 4.696 bpb (209% of gzip) | 4.356 bpb (234% of gzip) | 8.023 bpb (100% of gzip) |
| ctx4 | 1.739 bpb (77% of gzip) | 1.666 bpb (90% of gzip) | 8.420 bpb (105% of gzip) |
| ctx8 | 1.622 bpb (72% of gzip) | 1.616 bpb (87% of gzip) | 8.420 bpb (105% of gzip) |
| ctx | 1.430 bpb (64% of gzip) | 1.456 bpb (78% of gzip) | 8.136 bpb (102% of gzip) |
| gru | 1.935 bpb (86% of gzip) | 1.740 bpb (93% of gzip) | 8.045 bpb (101% of gzip) |
| nz | 1.319 bpb (59% of gzip) | 1.342 bpb (72% of gzip) | 8.002 bpb (100% of gzip) |
