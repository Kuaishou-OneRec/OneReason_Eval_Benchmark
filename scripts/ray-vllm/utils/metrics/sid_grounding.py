import torch


class Metrics(object):
    """Combined SID grounding metrics.

    - ``cov_rate``: fraction of unique prompt-history SIDs that appear in the CoT.
      Equals 1.0 when the prompt has no SIDs (vacuous).
    - ``sid_valid_rate``: fraction of CoT SID candidates that decode to a
      well-formed SID tuple (begin-tag + D codebook tokens with monotonic
      indices and in-range token ids). Equals 1.0 when the CoT emits no SID
      candidates (vacuous).

    Caller must populate:
      * ``cot_sids_tokens``: (N, D+1) LongTensor of valid CoT SIDs (begin-tag prefixed).
      * ``prompt_sids_tokens``: (M, D+1) LongTensor of valid prompt SIDs.
      * ``num_invalid_cot_sids``: int, count of SID-token windows in the CoT that
        failed validation.
      * ``num_cot_sids``: int, total count of SID-token positions in the CoT
        (valid + invalid * sids_length, as returned by
        ``LogitsSaverBuffer.extract_valid_sids``).
    """

    def __call__(self, config):
        cot_sids_tokens = config["cot_sids_tokens"]
        cot_sids_tokens = torch.unique(cot_sids_tokens, dim=0)              # N * D
        prompt_sids_tokens = config["prompt_sids_tokens"]
        prompt_sids_tokens = torch.unique(prompt_sids_tokens, dim=0)        # M * D

        total = prompt_sids_tokens.shape[0]
        if total > 0:
            matched = (cot_sids_tokens[None] == prompt_sids_tokens[:, None]).all(dim=-1).any(dim=0).float().sum().item()
            cov_rate = matched / total
        else:
            cov_rate = 1.0

        num_invalid = config.get("num_invalid_cot_sids", 0)
        num_total = config.get("num_cot_sids", 0)
        if num_total > 0:
            sid_valid_rate = 1.0 - (num_invalid / num_total)
        else:
            sid_valid_rate = 1.0

        return {
            "cov_rate": cov_rate,
            "sid_valid_rate": sid_valid_rate,
        }
