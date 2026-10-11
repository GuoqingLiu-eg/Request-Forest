from __future__ import annotations

import json


def observed_path_records(*, token_ids, parent_rows, depths, request_slots,
                          packed_to_node, node_values, target_predict,
                          request_ids=None, round_index=0):
    sizes = {len(x) for x in (token_ids, parent_rows, depths, request_slots, packed_to_node)}
    if len(sizes) != 1 or len(target_predict) < len(token_ids):
        raise ValueError("calibration row arrays disagree")
    matches = {}

    def accepted(row):
        if row in matches:
            return matches[row]
        if packed_to_node[row] < 0:
            matches[row] = True
        else:
            parent = parent_rows[row]
            if not 0 <= parent < row or request_slots[parent] != request_slots[row]:
                raise ValueError("calibration plan is not an ordered closed forest")
            matches[row] = accepted(parent) and token_ids[row] == target_predict[parent]
        return matches[row]

    best = {}
    for row, node in enumerate(packed_to_node):
        if node < 0:
            continue
        key = (request_slots[row], depths[row])
        score = node_values[node]
        if key not in best or score > node_values[packed_to_node[best[key]]]:
            best[key] = row
    records = []
    for (request, depth), row in sorted(best.items()):
        records.append({"depth": depth, "confidence": node_values[packed_to_node[row]],
                        "accepted": bool(accepted(row)), "split": "calibration",
                        "request_id": str(request) if request_ids is None else request_ids[request],
                        "round": round_index})
    return records


def append_observed_paths(path, plan, node_values, target_predict, *, round_index):
    logical = plan.logical_rows
    records = observed_path_records(
        token_ids=plan.token_ids[:logical].tolist(),
        parent_rows=plan.parent_rows[:logical].tolist(),
        depths=plan.depths[:logical].tolist(),
        request_slots=plan.request_slots[:logical].tolist(),
        packed_to_node=plan.packed_to_node[:logical].tolist(),
        node_values=node_values.tolist(),
        target_predict=target_predict.detach().cpu().tolist(),
        request_ids=[r.request_id for r in plan.request_refs], round_index=round_index,
    )
    with open(path, "a", encoding="utf-8") as stream:
        stream.writelines(json.dumps(record) + "\n" for record in records)
