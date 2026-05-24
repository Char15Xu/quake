#!/usr/bin/env python3
"""
LRU Partition Cache Experiment
==============================
Builds a 10M SPACEV1B index, uploads partitions to S3, then runs the same
query workload under different cache_capacity settings to measure:
  - Cache hit rate
  - S3 downloads per query
  - Per-query latency (total, S3 fetch, scan)
  - Recall (to confirm cache doesn't affect correctness)

Usage:
    # Full run (build + upload + benchmark):
    python cache_experiment.py

    # Skip build/upload if index already exists on S3:
    python cache_experiment.py --skip-build

    # Customize S3 settings via environment variables:
    QUAKE_S3_BUCKET=my-bucket QUAKE_S3_PREFIX=spacev10m python cache_experiment.py
"""

import argparse
import json
import os
import struct
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import quake
from quake.utils import compute_recall
from s3_utils import upload_index_to_s3

# ─── Configuration ───────────────────────────────────────────────────────────
DATASET_PATH = "/users/charlesx/SPTAG/datasets/SPACEV1B/vectors.bin/vectors_merged.bin"
QUERY_PATH   = "/users/charlesx/SPTAG/datasets/SPACEV1B/query.bin"
GT_PATH      = "/users/charlesx/quake/spacev10m_gt.bin"

S3_BUCKET   = os.environ.get("QUAKE_S3_BUCKET", "my-quake-bucket")
S3_PREFIX   = os.environ.get("QUAKE_S3_PREFIX", "spacev10m")
AWS_REGION  = os.environ.get("QUAKE_S3_REGION", "us-east-1")
S3_ENDPOINT = os.environ.get("QUAKE_S3_ENDPOINT", "")

INDEX_DIR    = "quake_spacev10m.index"
RESULTS_DIR  = Path("cache_experiment_results")

NUM_VECTORS  = 10_000_000
NLIST        = 1024
METRIC       = "l2"
NUM_QUERIES  = 1000        # queries per cache setting
TOP_K        = 10
RECALL_TARGET = 0.9

# Cache capacities to sweep (in number of partitions).
# Testing 0% (baseline), 10% (102), 15% (153), and 20% (204) of the 1024 total partitions.
CACHE_CAPACITIES = [0, 102, 153, 204]
# ─────────────────────────────────────────────────────────────────────────────


def load_queries():
    """Load SPACEV1B query vectors."""
    with open(QUERY_PATH, "rb") as f:
        n = struct.unpack("i", f.read(4))[0]
        d = struct.unpack("i", f.read(4))[0]
        data = np.frombuffer(f.read(n * d), dtype=np.int8).reshape(n, d)
    print(f"Loaded {n:,} queries, d={d}")
    return data


def load_ground_truth():
    """Load precomputed ground truth for the 10M subset."""
    with open(GT_PATH, "rb") as f:
        n = struct.unpack("i", f.read(4))[0]
        k = struct.unpack("i", f.read(4))[0]
        ids = np.frombuffer(f.read(n * k * 4), dtype=np.int32).reshape(n, k)
        # skip distances
    print(f"Loaded ground truth: {n:,} queries, k={k}")
    return ids


def build_and_upload():
    """Phase 1+2: Build 10M index and upload partitions to S3."""
    print(f"\n{'='*60}")
    print("Phase 1: Building index")
    print(f"{'='*60}")

    with open(DATASET_PATH, "rb") as f:
        total = struct.unpack("i", f.read(4))[0]
        d = struct.unpack("i", f.read(4))[0]

    n = min(total, NUM_VECTORS)
    dataset = np.memmap(DATASET_PATH, dtype=np.int8, mode="r", offset=8, shape=(n, d))
    vectors = torch.from_numpy(dataset[:n].copy()).to(torch.float32)
    ids = torch.arange(n, dtype=torch.int64)

    index = quake.QuakeIndex()
    build_params = quake.IndexBuildParams()
    build_params.nlist = NLIST
    build_params.metric = METRIC

    t0 = time.time()
    index.build(vectors, ids, build_params)
    print(f"Build time: {time.time() - t0:.1f}s")

    index.save(INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}")

    print(f"\n{'='*60}")
    print("Phase 2: Uploading partitions to S3")
    print(f"{'='*60}")
    upload_index_to_s3(
        INDEX_DIR, S3_BUCKET, S3_PREFIX,
        region=AWS_REGION,
        endpoint_url=S3_ENDPOINT if S3_ENDPOINT else None,
    )


def run_queries(cache_capacity, queries, gt_ids):
    """
    Load the index with the given cache_capacity, run NUM_QUERIES queries,
    and return per-query statistics.
    """
    print(f"\n--- cache_capacity = {cache_capacity} ---")

    # Load index fresh for each cache setting
    index = quake.QuakeIndex()
    index.load(
        INDEX_DIR,
        s3_bucket=S3_BUCKET,
        s3_prefix=S3_PREFIX,
        s3_region=AWS_REGION,
        s3_endpoint=S3_ENDPOINT,
        cache_capacity=cache_capacity,
    )
    print(f"  Index loaded (nlist={index.nlist()}, ntotal={index.ntotal()})")

    # Warmup: run 5 queries to prime the system
    last_warmup_ti = None
    for i in range(min(5, NUM_QUERIES)):
        q = torch.from_numpy(queries[i].copy()).to(torch.float32).reshape(1, -1)
        sp = quake.SearchParams()
        sp.k = TOP_K
        sp.recall_target = RECALL_TARGET
        res = index.search(q, sp)
        last_warmup_ti = res.timing_info

    # Collect per-query stats
    prev_s3_load_ms = last_warmup_ti.s3_load_time_ns / 1e6 if last_warmup_ti else 0.0
    prev_n_s3_downloads = last_warmup_ti.n_s3_downloads if last_warmup_ti else 0
    prev_cache_hits = last_warmup_ti.cache_hits if last_warmup_ti else 0
    prev_cache_misses = last_warmup_ti.cache_misses if last_warmup_ti else 0
    
    prev_worker_lookup_ms = 0
    prev_enqueue_ms = 0
    prev_manager_queue_wait_ms = 0
    prev_manager_s3_wait_ms = 0
    prev_bg_update_ms = 0
    prev_bg_evict_ms = 0
    
    latency_ms = []
    s3_load_ms = []
    scan_ms = []
    n_s3_downloads = []
    cache_hits = []
    cache_misses = []
    recalls = []
    
    worker_lookup_ms = []
    enqueue_ms = []
    manager_queue_wait_ms = []
    manager_s3_wait_ms = []
    bg_update_ms = []
    bg_evict_ms = []
    
    partitions_scanned = []

    nq = min(NUM_QUERIES, len(queries))
    for i in range(nq):
        q = torch.from_numpy(queries[i].copy()).to(torch.float32).reshape(1, -1)
        sp = quake.SearchParams()
        sp.k = TOP_K
        sp.recall_target = RECALL_TARGET
        sp.s3_prefetch_initial = 0
        sp.s3_prefetch_lookahead = 0

        t0 = time.perf_counter()
        result = index.search(q, sp)
        elapsed_ms = (time.perf_counter() - t0) * 1e3

        ti = result.timing_info
        latency_ms.append(elapsed_ms)
        scan_ms.append(ti.scan_time_ns / 1e6)
        partitions_scanned.append(ti.partitions_scanned)

        if cache_capacity > 0:
            cur_s3_load_ms = ti.s3_load_time_ns / 1e6
            cur_n_s3_downloads = ti.n_s3_downloads
            cur_cache_hits = ti.cache_hits
            cur_cache_misses = ti.cache_misses
            
            cur_worker_lookup_ms = ti.cache_worker_lookup_ns / 1e6
            cur_enqueue_ms = ti.cache_enqueue_ns / 1e6
            cur_manager_queue_wait_ms = ti.cache_manager_queue_wait_ns / 1e6
            cur_manager_s3_wait_ms = ti.cache_manager_s3_wait_ns / 1e6
            cur_bg_update_ms = ti.cache_bg_process_update_ns / 1e6
            cur_bg_evict_ms = ti.cache_bg_evict_ns / 1e6

            s3_load_ms.append(cur_s3_load_ms - prev_s3_load_ms)
            n_s3_downloads.append(cur_n_s3_downloads - prev_n_s3_downloads)
            cache_hits.append(cur_cache_hits - prev_cache_hits)
            cache_misses.append(cur_cache_misses - prev_cache_misses)
            
            worker_lookup_ms.append(cur_worker_lookup_ms - prev_worker_lookup_ms)
            enqueue_ms.append(cur_enqueue_ms - prev_enqueue_ms)
            manager_queue_wait_ms.append(cur_manager_queue_wait_ms - prev_manager_queue_wait_ms)
            manager_s3_wait_ms.append(cur_manager_s3_wait_ms - prev_manager_s3_wait_ms)
            bg_update_ms.append(cur_bg_update_ms - prev_bg_update_ms)
            bg_evict_ms.append(cur_bg_evict_ms - prev_bg_evict_ms)

            prev_s3_load_ms = cur_s3_load_ms
            prev_n_s3_downloads = cur_n_s3_downloads
            prev_cache_hits = cur_cache_hits
            prev_cache_misses = cur_cache_misses
            
            prev_worker_lookup_ms = cur_worker_lookup_ms
            prev_enqueue_ms = cur_enqueue_ms
            prev_manager_queue_wait_ms = cur_manager_queue_wait_ms
            prev_manager_s3_wait_ms = cur_manager_s3_wait_ms
            prev_bg_update_ms = cur_bg_update_ms
            prev_bg_evict_ms = cur_bg_evict_ms
        else:
            s3_load_ms.append(ti.s3_load_time_ns / 1e6)
            n_s3_downloads.append(ti.n_s3_downloads)
            cache_hits.append(ti.cache_hits)
            cache_misses.append(ti.cache_misses)
            
            worker_lookup_ms.append(0.0)
            enqueue_ms.append(0.0)
            manager_queue_wait_ms.append(0.0)
            manager_s3_wait_ms.append(0.0)
            bg_update_ms.append(0.0)
            bg_evict_ms.append(0.0)
        recalls.append(compute_recall(result.ids, gt_ids[i].reshape(1, -1), TOP_K))

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{nq} queries done\r", end="")

    print(f"  {nq}/{nq} queries done")

    return {
        "latency_ms": latency_ms,
        "s3_load_ms": s3_load_ms,
        "worker_lookup_ms": worker_lookup_ms,
        "enqueue_ms": enqueue_ms,
        "manager_queue_wait_ms": manager_queue_wait_ms,
        "manager_s3_wait_ms": manager_s3_wait_ms,
        "bg_update_ms": bg_update_ms,
        "bg_evict_ms": bg_evict_ms,
        "scan_ms": scan_ms,
        "n_s3_downloads": n_s3_downloads,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "recalls": recalls,
        "partitions_scanned": partitions_scanned,
    }


def summarize(stats):
    """Compute summary statistics from per-query lists."""
    def pct(arr, ps=[0, 10, 50, 90, 99, 100]):
        return {f"p{p}": round(float(np.percentile(arr, p)), 3) for p in ps}

    total_hits = sum(stats["cache_hits"])
    total_misses = sum(stats["cache_misses"])
    total_accesses = total_hits + total_misses
    hit_rate = total_hits / total_accesses if total_accesses > 0 else 0.0

    return {
        "num_queries": len(stats["latency_ms"]),
        "recall_avg": round(float(np.mean(stats["recalls"])), 4),
        "cache_hit_rate": round(hit_rate, 4),
        "cache_total_hits": int(total_hits),
        "cache_total_misses": int(total_misses),
        "latency_ms": {"avg": round(float(np.mean(stats["latency_ms"])), 3), **pct(stats["latency_ms"])},
        "s3_load_ms": {"avg": round(float(np.mean(stats["s3_load_ms"])), 3), **pct(stats["s3_load_ms"])},
        "worker_lookup_ms": {"avg": round(float(np.mean(stats["worker_lookup_ms"])), 3), **pct(stats["worker_lookup_ms"])},
        "enqueue_ms": {"avg": round(float(np.mean(stats["enqueue_ms"])), 3), **pct(stats["enqueue_ms"])},
        "manager_queue_wait_ms": {"avg": round(float(np.mean(stats["manager_queue_wait_ms"])), 3), **pct(stats["manager_queue_wait_ms"])},
        "manager_s3_wait_ms": {"avg": round(float(np.mean(stats["manager_s3_wait_ms"])), 3), **pct(stats["manager_s3_wait_ms"])},
        "bg_update_ms": {"avg": round(float(np.mean(stats["bg_update_ms"])), 3), **pct(stats["bg_update_ms"])},
        "bg_evict_ms": {"avg": round(float(np.mean(stats["bg_evict_ms"])), 3), **pct(stats["bg_evict_ms"])},
        "scan_ms": {"avg": round(float(np.mean(stats["scan_ms"])), 3), **pct(stats["scan_ms"])},
        "s3_downloads_per_query": {"avg": round(float(np.mean(stats["n_s3_downloads"])), 2), **pct(stats["n_s3_downloads"])},
        "partitions_scanned": {"avg": round(float(np.mean(stats["partitions_scanned"])), 2), **pct(stats["partitions_scanned"])},
    }


def print_summary_table(results):
    """Print a compact comparison table across cache sizes."""
    print(f"\n{'='*90}")
    print("CACHE EXPERIMENT RESULTS")
    print(f"{'='*90}")
    header = f"{'Cache Cap':>10} | {'Hit Rate':>8} | {'Avg Lat(ms)':>11} | {'Lkp+Enq(ms)':>11} | {'Q Wait(ms)':>10} | {'S3 Wait(ms)':>11} | {'Avg S3 DL':>9} | {'Raw S3(ms)':>10} | {'Recall':>7}"
    print(header)
    print("-" * len(header))

    for cap in CACHE_CAPACITIES:
        key = str(cap)
        if key not in results:
            continue
        v = results[key]
        hr = v["cache_hit_rate"]
        lat = v["latency_ms"]["avg"]
        s3w = v["manager_s3_wait_ms"]["avg"]
        qw = v["manager_queue_wait_ms"]["avg"]
        lkp = v["worker_lookup_ms"]["avg"] + v["enqueue_ms"]["avg"]
        raw = v["s3_load_ms"]["avg"]
        s3d = v["s3_downloads_per_query"]["avg"]
        rec = v["recall_avg"]
        print(f"{key:>10} | {hr:>8.3f} | {lat:>11.2f} | {lkp:>11.3f} | {qw:>10.2f} | {s3w:>11.2f} | {s3d:>9.1f} | {raw:>10.2f} | {rec:>7.4f}")

    print(f"{'='*90}")


def main():
    global NUM_QUERIES, CACHE_CAPACITIES

    parser = argparse.ArgumentParser(description="LRU Cache Experiment")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip index build and S3 upload.")
    parser.add_argument("--num-queries", type=int, default=NUM_QUERIES,
                        help=f"Number of queries per cache setting (default: {NUM_QUERIES}).")
    parser.add_argument("--cache-sizes", type=int, nargs="+", default=None,
                        help="Override cache capacity list (e.g. --cache-sizes 0 128 512).")
    args = parser.parse_args()

    NUM_QUERIES = args.num_queries
    if args.cache_sizes:
        CACHE_CAPACITIES = sorted(args.cache_sizes)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1+2: Build and upload
    if not args.skip_build:
        build_and_upload()
    else:
        print("Skipping index build and S3 upload.")

    # Load queries and ground truth
    queries = load_queries()
    gt_ids = load_ground_truth()

    # Phase 3: Run queries for each cache capacity
    print(f"\n{'='*60}")
    print(f"Phase 3: Running {NUM_QUERIES} queries per cache setting")
    print(f"Cache capacities: {CACHE_CAPACITIES}")
    print(f"{'='*60}")

    all_results = {}
    for cap in CACHE_CAPACITIES:
        stats = run_queries(cap, queries, gt_ids)
        all_results[str(cap)] = summarize(stats)

    # Print comparison table
    print_summary_table(all_results)

    # Save full results to JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"cache_experiment_{ts}.json"
    payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": "SPACEV1B_10M",
            "nlist": NLIST,
            "num_queries": NUM_QUERIES,
            "top_k": TOP_K,
            "recall_target": RECALL_TARGET,
            "cache_capacities": CACHE_CAPACITIES,
            "s3_bucket": S3_BUCKET,
            "s3_prefix": S3_PREFIX,
        },
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
