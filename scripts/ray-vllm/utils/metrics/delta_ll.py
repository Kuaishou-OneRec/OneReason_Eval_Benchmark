import numpy as np


class Metrics(object):

    def __call__(self, config):
        cot = np.asarray(config["cot_prompt_logprobs"], dtype=np.float64)
        base = np.asarray(config["prompt_logprobs"], dtype=np.float64)
        # Per-SID logprobs; mean over the N ground-truth SIDs for this sample.
        if cot.size == 0 or base.size == 0 or cot.shape != base.shape:
            return {"delta_ll": 0.0, "delta_ll_per_sid": []}
        diff = cot - base
        return {
            "delta_ll": float(diff.mean()),
            "delta_ll_per_sid": diff.tolist(),
        }
