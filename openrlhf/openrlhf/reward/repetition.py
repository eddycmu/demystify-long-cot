import logging
from typing import List
import torch
from openrlhf.reward.common import DenseReward


logger = logging.getLogger(__name__)


class RepetitionDensePenalty(DenseReward):
    def __init__(self, ngram_size: int, penalty: float, only_start: bool):
        if penalty >= 0:
            raise ValueError(f"Expected penalty to be negative, instead got {penalty}")

        self._ngram_size = ngram_size
        self._penalty = penalty
        self._only_start = only_start

        logger.info(
            "Initialized repetition dense penalty with"
            f" ngram_size: {ngram_size}, penalty: {penalty}, only_start: {only_start}")

    @staticmethod
    def is_selected(remote_rm_url: str) -> bool:
        return remote_rm_url == "math_rule_repetition_dense"

    def reward(self, sequences: List[str], gen_lengths: List[int], answers: List[str], scores: List[float], output_ids: List[List[int]], num_actions: int) -> torch.Tensor:
        assert len(sequences) == len(gen_lengths)
        assert len(sequences) == len(answers)
        assert len(sequences) == len(output_ids)

        rewards = []
        for out, out_len in zip(output_ids, gen_lengths):
            gen = out[:int(out_len)]
            repeated = []
            ngrams = set()
            for start_idx, ng in enumerate(zipngram_tokens(gen, self._ngram_size)):
                if ng in ngrams:
                    repeated.append(start_idx)
                ngrams.add(ng)

            curr_reward = [0] * num_actions
            curr_end_idx = -1
            for start_idx in repeated:
                if not self._only_start or start_idx > curr_end_idx:
                    for i in range(start_idx, start_idx + self._ngram_size):
                        curr_reward[i] = self._penalty

                curr_end_idx = start_idx + self._ngram_size

            rewards.append(curr_reward)

        return torch.tensor(rewards)


def get_repetition_penalty(ngram_size: int, max_penalty: float, generation: str) -> float:
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if max_penalty == 0:
        return 0

    ngrams = set()
    total = 0
    for ng in zipngram(generation, ngram_size):
        ngrams.add(ng)
        total += 1

    scaling = 1 - len(ngrams) / total
    return scaling * max_penalty


# Source:
# https://stackoverflow.com/questions/21883108/fast-optimize-n-gram-implementations-in-python
def zipngram(text: str, ngram_size: int):
    words = text.lower().split()
    return zip(*[words[i:] for i in range(ngram_size)])


def zipngram_tokens(tokens: List[int], ngram_size: int):
    return zip(*[tokens[i:] for i in range(ngram_size)])