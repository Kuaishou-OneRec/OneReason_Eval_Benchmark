import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import bisect
from typing import Optional, List, Tuple


class LogitsSaverBuffer(object):

    def __init__(
        self,
        temperature: float = 0.7,
        think_start_id: int = 151667,
        think_end_id: int = 151668,
        sids_token_ids: Optional[torch.LongTensor] = None,
        sids_length: int = 3,
        sids_begin_ids: Optional[torch.LongTensor] = None,
        prompt_token_id: int = None,
        valid_sids: Optional[List[Tuple[int]]] = None,
    ):
        self.temperature = temperature
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.prompt_token_id = prompt_token_id

        self.buffer = list()
        self.sids_token_ids = sids_token_ids
        self.sids_begin_ids = sids_begin_ids
        self.sids_length = sids_length
        self.valid_sids = valid_sids

        chunks = sids_token_ids.reshape(sids_length, -1)    # D * S
        self.sids_bounds = [chunks.min(dim=1)[0], chunks.max(dim=1)[0]]

    def append(self, logits):
        if isinstance(logits, np.ndarray):
            logits = torch.tensor(logits)
        logits = logits.float().flatten().cpu()
        self.buffer.append(logits)

    def exclude_invalid_sids(self, tokens, indices, orig_tokens):
        lower_bound, upper_bound = self.sids_bounds
        sids_length = self.sids_length
        length = tokens.shape[0]
        if length < sids_length:
            return length, length, tokens[:0], indices[:0]
        windows_tokens = tokens.unfold(0, sids_length, 1)  # L * D
        windows_indices = indices.unfold(0, sids_length, 1) # L * D
        mask = torch.logical_and((windows_tokens >= lower_bound).all(dim=1), (windows_tokens <= upper_bound).all(dim=1))
        diff = windows_indices[:, 1:] - windows_indices[:, :-1]
        mask = torch.logical_and(mask, (diff == 1).all(dim=1))
        begin_tag_indices = windows_indices[:, 0] - 1
        mask = torch.logical_and(mask, begin_tag_indices >= 0)
        begin_tag_indices[begin_tag_indices < 0] = 0
        begin_tokens = orig_tokens[begin_tag_indices]
        begin_mask = (begin_tokens[:, None] == self.sids_begin_ids).any(dim=1)
        mask = torch.logical_and(mask, begin_mask)

        valid_tokens = windows_tokens[mask].flatten().reshape(-1, sids_length)
        valid_indices = windows_indices[mask].flatten().reshape(-1, sids_length)
        num_invalid = length - valid_tokens.numel()

        valid_tokens = torch.concat(
            [
                begin_tokens[mask].unsqueeze(1),
                valid_tokens
            ],
            dim=1
        ).reshape(-1, sids_length + 1)
        valid_indices = torch.concat(
            [
                begin_tag_indices[mask].unsqueeze(1),
                valid_indices
            ],
            dim=1
        ).flatten()
        indices_mask = orig_tokens.new_ones(size=orig_tokens.shape, dtype=torch.bool)
        indices_mask[valid_indices] = False
        nl_tokens = orig_tokens[indices_mask]
        return num_invalid, length, valid_tokens, nl_tokens

    def extract_valid_sids(self, orig_tokens, mask):
        indices = mask.nonzero().flatten()
        tokens = orig_tokens[mask]
        return self.exclude_invalid_sids(tokens, indices, orig_tokens)

    def extract_prompt_sids(self, prompt_tokens):
        if isinstance(prompt_tokens, list):
            prompt_tokens = torch.LongTensor(prompt_tokens)

        mask = (prompt_tokens[:, None] == self.sids_token_ids).any(-1)
        _, _, prompt_sids_tokens, _ = self.extract_valid_sids(prompt_tokens, mask)
        return prompt_sids_tokens

    def collect_cot(
        self,
        prompt_tokens: List[int],
        generate_tokens: List[int],
        orig_prompt_tokens: List[int],
    ):
        # entropy = \sum_i p_i\log p_i = \sum_i p_i\log (\exp(z_i)/ \sum_j \exp(z_j))
        # = \sum_i p_i(\log \exp(z_i)) - \sum_i p_i \log \sum_j \exp(z_j))
        # = \sum_i p_i * z_i - \log \sum_j \exp(z_j)
        if len(self.buffer) > 0:
            logits = torch.stack(self.buffer, dim=0)
        else:
            # No per-step logits collected (memory-frugal stage-1 path used
            # for ehr/cov_rate/sid_valid_rate which do not need them).
            logits = None

        if not isinstance(generate_tokens, torch.Tensor):
            generate_tokens = torch.LongTensor(generate_tokens).flatten()

        if logits is not None:
            assert logits.shape[0] == generate_tokens.shape[0]

        think_start_mask = (generate_tokens == self.think_start_id)
        think_end_mask = (generate_tokens == self.think_end_id)

        if think_start_mask.any():
            cot_start = torch.argmax(think_start_mask.float()).item() + 1
        else:
            cot_start = 0

        if think_end_mask.any():
            cot_end = torch.argmax(think_end_mask.float()).item()
        else:
            cot_end = think_end_mask.shape[0]

        sids_mask = (generate_tokens.unsqueeze(-1) == self.sids_token_ids).any(dim=-1)
        cot_tokens = generate_tokens[cot_start: cot_end].clone()
        cot_sids_mask = sids_mask[cot_start: cot_end].clone()

        num_invalid_cot_sids, num_cot_sids, cot_sids_tokens, cot_nl_tokens = self.extract_valid_sids(cot_tokens, cot_sids_mask)
        cot_sids_tokens = cot_sids_tokens.reshape(-1, self.sids_length + 1)

        prompt_sids_tokens = self.extract_prompt_sids(orig_prompt_tokens)

        if logits is not None:
            logits_div_temp = logits / self.temperature
            probs = F.softmax(logits, dim=-1)
            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
            cot_nl_entropy = entropy[cot_start: cot_end][~cot_sids_mask]
        else:
            logits_div_temp = None
            probs = None
            entropy = None
            cot_nl_entropy = None

        # 找到answer(不含cot)里第一个非sid的位置
        answer_sids_mask = sids_mask.clone()
        answer_sids_mask[:cot_end + 1] = False

        if sids_mask.any():
            answer_start = torch.argmax(answer_sids_mask.float()).item() - 1    # 去掉<|XXX_begin|>
        else:
            answer_start = answer_sids_mask.shape[0] - 1

        if think_end_mask.any():
            prompt_with_cot_tokens = prompt_tokens + generate_tokens[:answer_start].tolist()
        else:
            prompt_with_cot_tokens = prompt_tokens + generate_tokens.tolist()

        prompt_with_sid_tokens = prompt_tokens + generate_tokens[:cot_start].tolist() + cot_sids_tokens.flatten().tolist() + generate_tokens[cot_end:answer_start].tolist()
        prompt_with_nl_tokens = prompt_tokens + generate_tokens[:cot_start].tolist() + cot_nl_tokens.flatten().tolist() + generate_tokens[cot_end:answer_start].tolist()

        if self.prompt_token_id is not None:
            prompt_with_cot_tokens.append(self.prompt_token_id)
            prompt_with_sid_tokens.append(self.prompt_token_id)
            prompt_with_nl_tokens.append(self.prompt_token_id)
            prompt_tokens.append(self.prompt_token_id)
            orig_prompt_tokens.append(self.prompt_token_id)

        if self.valid_sids is not None:
            hitted_cot_sids = 0
            total_cot_sids = 0
            cot_sids_without_begin = cot_sids_tokens[:, 1:].tolist()
            for sids in cot_sids_without_begin:
                sids = tuple(sids)
                pos = bisect.bisect_left(self.valid_sids, sids)
                if (0 <= pos < len(self.valid_sids)) and (self.valid_sids[pos] == sids):
                    hitted_cot_sids += 1
                total_cot_sids += 1
            hitted_cot_sids_in_library = (hitted_cot_sids / total_cot_sids) if total_cot_sids > 0 else 1.0
        else:
            hitted_cot_sids_in_library = None

        return {
            "entropy": entropy,
            "logits": logits,
            "logits_div_temp": logits_div_temp,
            "probs": probs,
            "generate_tokens": generate_tokens,
            "cot_tokens": cot_tokens,
            "cot_sids_tokens": cot_sids_tokens,
            "cot_nl_entropy": cot_nl_entropy,
            "prompt_with_nl_tokens": prompt_with_nl_tokens,
            "prompt_with_cot_tokens": prompt_with_cot_tokens,
            "prompt_with_sid_tokens": prompt_with_sid_tokens,
            "prompt_tokens": prompt_tokens,
            "prompt_sids_tokens": prompt_sids_tokens,
            "num_invalid_cot_sids": num_invalid_cot_sids,
            "num_cot_sids": num_cot_sids,
            "orig_prompt_tokens": orig_prompt_tokens,
            "hitted_cot_sids_in_library": hitted_cot_sids_in_library,
        }

    def collect_target(self, target_sids):
        if isinstance(target_sids, list):
            target_sids = torch.LongTensor(target_sids)

        if target_sids.ndim != 1:
            raise ValueError("`target_sids` must be a 1d tensor.")

        logits = torch.stack(self.buffer, dim=0)

        if logits.shape[0] != target_sids.shape[0]:
            raise ValueError("Length of `logits` must equals to target_sids's.")

        logprobs = F.log_softmax(logits, dim=-1)
        indices = torch.arange(logits.shape[0], device=logits.device, dtype=torch.int64)
        cum_logprobs = logprobs[indices, target_sids].sum().item()

        return cum_logprobs
