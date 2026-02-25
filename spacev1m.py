import quake
from quake.utils import compute_recall
import torch

import struct
import numpy as np
import os
import time

print("Loading dataset...")
fdataset = open('/users/yuhong/nvme1n1/SPTAG/datasets/SPACEV1B/vectors.bin/vectors_merged.bin', 'rb')
dataset_count = struct.unpack('i', fdataset.read(4))[0]
dataset_count = min(dataset_count, 10000000)
dataset_dimension = struct.unpack('i', fdataset.read(4))[0]
dataset = np.frombuffer(fdataset.read(dataset_count * dataset_dimension), dtype=np.int8).reshape((dataset_count, dataset_dimension))

print("Loading queries...")
fq = open('/users/yuhong/nvme1n1/SPTAG/datasets/SPACEV1B/query.bin', 'rb')
q_count = struct.unpack('i', fq.read(4))[0]
q_dimension = struct.unpack('i', fq.read(4))[0]
queries = np.frombuffer(fq.read(q_count * q_dimension), dtype=np.int8).reshape((q_count, q_dimension))

print("Loading truth...")
ftruth = open('/users/yuhong/nvme1n1/SPTAG/datasets/SPACEV1B/gt_10m.bin', 'rb')
t_count = struct.unpack('i', ftruth.read(4))[0]
topk = struct.unpack('i', ftruth.read(4))[0]
truth_vids = np.frombuffer(ftruth.read(t_count * topk * 4), dtype=np.int32).reshape((t_count, topk))
truth_distances = np.frombuffer(ftruth.read(t_count * topk * 4), dtype=np.float32).reshape((t_count, topk))

vectors = torch.from_numpy(dataset.copy()).to(torch.float32)
ids = torch.arange(dataset_count)

index = quake.QuakeIndex()
build_params = quake.IndexBuildParams()
build_params.nlist = 1024  # Number of clusters
build_params.metric = "l2" # Use Euclidean distance
start_time = time.time()
index.build(vectors, ids, build_params)
end_time = time.time()
print("Build time:", end_time - start_time)

index.save("quake_spacev10m.index")
print("Index saved")

recall_list = list()
latency_list = list()
scanned_partitions_list = list()

recall_target = 0.9

for top_K in [10, 30, 50, 100]:
    for i in range(q_count):
        query = torch.from_numpy(queries[i].copy()).to(torch.float32).reshape(1, -1)
        search_params = quake.SearchParams()
        search_params.k = top_K
        # search_params.nprobe = 10
        search_params.recall_target = recall_target
        start_time = time.perf_counter()
        search_result = index.search(query, search_params)
        end_time = time.perf_counter()
        latency_list.append((end_time - start_time) * 1e3)
        recall_list.append(compute_recall(search_result.ids, truth_vids[i].reshape(1, -1), top_K))
        scanned_partitions_list.append(search_result.timing_info.partitions_scanned)
        print(f"{i + 1}/{q_count}\r", end="")
    print()

    print(f"Recall target: {recall_target}, top K: {top_K}")
    print("Recall: avg {:.2f}, p0 {:.2f}, p10 {:.2f}, p20 {:.2f}, p30 {:.2f}, p40 {:.2f}, "
        "p50 {:.2f}, p60 {:.2f}, p70 {:.2f}, p80 {:.2f}, p90 {:.2f}, p99 {:.2f}, p100 {:.2f}".format(
            np.mean(recall_list),
            np.percentile(recall_list, 0), np.percentile(recall_list, 10),
            np.percentile(recall_list, 20), np.percentile(recall_list, 30),
            np.percentile(recall_list, 40), np.percentile(recall_list, 50),
            np.percentile(recall_list, 60), np.percentile(recall_list, 70),
            np.percentile(recall_list, 80), np.percentile(recall_list, 90),
            np.percentile(recall_list, 99), np.percentile(recall_list, 100)))
    print("Latency: avg {:.2f} ms, p0 {:.2f} ms, p10 {:.2f} ms, p20 {:.2f} ms, p30 {:.2f} ms, "
        "p40 {:.2f} ms, p50 {:.2f} ms, p60 {:.2f} ms, p70 {:.2f} ms, p80 {:.2f} ms, p90 {:.2f} ms, p99 {:.2f} ms, p100 {:.2f} ms".format(
            np.mean(latency_list),
            np.percentile(latency_list, 0), np.percentile(latency_list, 10),
            np.percentile(latency_list, 20), np.percentile(latency_list, 30),
            np.percentile(latency_list, 40), np.percentile(latency_list, 50),
            np.percentile(latency_list, 60), np.percentile(latency_list, 70),
            np.percentile(latency_list, 80), np.percentile(latency_list, 90),
            np.percentile(latency_list, 99), np.percentile(latency_list, 100)))
    print("Scanned partitions: avg {:.2f}, p0 {:.2f}, p10 {:.2f}, p20 {:.2f}, p30 {:.2f}, p40 {:.2f}, "
        "p50 {:.2f}, p60 {:.2f}, p70 {:.2f}, p80 {:.2f}, p90 {:.2f}, p99 {:.2f}, p100 {:.2f}".format(
            np.mean(scanned_partitions_list),
            np.percentile(scanned_partitions_list, 0), np.percentile(scanned_partitions_list, 10),
            np.percentile(scanned_partitions_list, 20), np.percentile(scanned_partitions_list, 30),
            np.percentile(scanned_partitions_list, 40), np.percentile(scanned_partitions_list, 50),
            np.percentile(scanned_partitions_list, 60), np.percentile(scanned_partitions_list, 70),
            np.percentile(scanned_partitions_list, 80), np.percentile(scanned_partitions_list, 90),
            np.percentile(scanned_partitions_list, 99), np.percentile(scanned_partitions_list, 100)))
