import torch


class Metrics(object):

    def __call__(self, config):
        cot_sids_tokens = config["cot_sids_tokens"]
        cot_sids_tokens = torch.unique(cot_sids_tokens, dim=0) # N * D
        prompt_sids_tokens = config["prompt_sids_tokens"]
        prompt_sids_tokens = torch.unique(prompt_sids_tokens, dim=0)    # M * D

        total, length = cot_sids_tokens.shape

        if total == 0:
            return {
                "ehr": 0.0,
            }

        matched = (prompt_sids_tokens[:, None] == cot_sids_tokens[None]).all(dim=-1).any(dim=0).float().sum().item()   # M * N

        return {
            "ehr": 1. - (matched / total)
        }
