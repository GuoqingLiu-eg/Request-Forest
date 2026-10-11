import json
import logging

import torch

logger = logging.getLogger(__name__)


class AdaptiveRFDiagnostics:
    def __init__(self, worker):
        self.worker = worker
        self.last_mode = None
        self.saved = None
        self.checked_transitions = 0

    def snapshot(self, batch, prepared):
        mode = prepared.decision.mode
        self.saved = None
        if self.last_mode is not None and self.last_mode != mode:
            table = batch.req_to_token_pool.req_to_token
            mapping = [table[row, :length].clone()
                       for row, length in zip(prepared.request_rows, prepared.committed_lens)]
            slots = torch.cat(mapping).long()
            caches = []
            for name, runner in [('target', self.worker.target_worker.model_runner),
                                 ('draft', self.worker.draft_model_runner)]:
                pool = runner.token_to_kv_pool
                for layer in range(pool.start_layer, pool.start_layer + len(pool.k_buffer)):
                    for component in ['key', 'value']:
                        getter = getattr(pool, 'get_' + component + '_buffer')
                        caches.append((name, layer, component, getter, getter(layer)[slots].clone()))
            self.saved = (mapping, slots, caches)
        self.last_mode = mode

    def finish(self, batch, prepared, result):
        if self.saved is None:
            return
        mapping, slots, caches = self.saved
        table = batch.req_to_token_pool.req_to_token
        for row, length, old in zip(prepared.request_rows, prepared.committed_lens, mapping):
            if not torch.equal(table[row, :length], old):
                raise RuntimeError('Adaptive switch changed a committed prefix mapping')
        for name, layer, component, getter, old in caches:
            if not torch.equal(getter(layer)[slots], old):
                raise RuntimeError(f'Adaptive switch changed committed {name} KV {layer}/{component}')
        self.checked_transitions += 1
        logger.info('Adaptive RF diagnostic %s', json.dumps(dict(
            event='prefix_unchanged', mode=prepared.decision.mode,
            batch=len(prepared.requests), checked_transitions=self.checked_transitions,
            committed_tokens=slots.numel(), kv_layer_components=len(caches),
        )))
        self.saved = None

    def trace_result(self, prepared, result, accepted_rows=None):
        index = 65
        logits = result.logits_output.next_token_logits
        stride = result.speculative_num_draft_tokens
        counts = result.accept_lens.tolist()
        output = result.next_token_ids.reshape(len(prepared.requests), stride)
        for i, req in enumerate(prepared.requests):
            j = index - len(req.output_ids)
            if not 0 <= j < counts[i]:
                continue
            row = i * self.worker.block_size + j if accepted_rows is None else int(accepted_rows[i, j])
            values, ids = torch.topk(logits[row].float(), 5)
            logger.info('Adaptive RF diagnostic %s', json.dumps(dict(
                event='token_logits', mode=prepared.decision.mode, batch=len(prepared.requests),
                request=i, output_index=index, local_output_index=j,
                prefix_tokens=list(req.origin_input_ids) + list(req.output_ids) + output[i, :j].tolist(),
                chosen_token=int(output[i, j]), target_argmax=int(logits[row].argmax()),
                top5_ids=ids.tolist(), top5_logits=values.tolist(),
                margin=float(values[0] - values[1]), comma_logit=float(logits[row, 11]),
                backtick_logit=float(logits[row, 1565]), verify_rows=logits.shape[0],
            )))
