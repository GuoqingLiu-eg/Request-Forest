class AdaptiveRFMetrics:


    def __init__(self):
        modes = ("dflash2", "rf")
        self.round_counts = dict.fromkeys(modes, 0)
        self.graph_hits = {mode: {"target": 0, "draft": 0} for mode in modes}
        self.active_batch_sum = dict.fromkeys(modes, 0)
        self.active_batch_histogram = {mode: {} for mode in modes}
        self.switch_count = 0
        self.switch_us_total = 0.0

    def record_round(
        self, *, mode, active_batch, target_graph, draft_graph, lane_assignment_us
    ):
        self.round_counts[mode] += 1
        self.graph_hits[mode]["target"] += int(target_graph)
        self.graph_hits[mode]["draft"] += int(draft_graph)
        self.active_batch_sum[mode] += active_batch
        histogram = self.active_batch_histogram[mode]

        bucket = str(active_batch)
        histogram[bucket] = histogram.get(bucket, 0) + 1
        if lane_assignment_us > 0:
            self.switch_count += 1
            self.switch_us_total += lane_assignment_us

    def snapshot(self):
        return {
            "round_counts": self.round_counts.copy(),
            "graph_hits": {mode: hits.copy() for mode, hits in self.graph_hits.items()},
            "active_batch_sum": self.active_batch_sum.copy(),
            "active_batch_histogram": {
                mode: counts.copy()
                for mode, counts in self.active_batch_histogram.items()
            },
            "switch_count": self.switch_count,
            "switch_us_total": self.switch_us_total,
            "switch_timing_scope": "cpu_lane_assignment_only",
            "counter_scope": "completed_nonempty_decode_rounds",
        }
