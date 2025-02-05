import logging
import math
from typing import Callable, List, Optional
import torch

from openrlhf.reward.common import MAX_LEN_MARGIN, SparseReward
from openrlhf.reward.repetition import get_repetition_penalty


logger = logging.getLogger(__name__)


class CosineScaledSparseReward(SparseReward):
    def __init__(
            self,
            min_value_wrong: float,
            max_value_wrong: float,
            min_value_correct: float,
            max_value_correct: float,
            max_len: int,
            exceed_length: Optional[float],
            repetition_max_penalty: float,
            repetition_ngram_size: int,
            score_fn: Callable[[str, str], float]):
        self.min_value_wrong = min_value_wrong
        self.min_value_correct = min_value_correct
        self.max_value_wrong = max_value_wrong
        self.max_value_correct = max_value_correct
        self.max_len = max_len
        self.exceed_length = exceed_length if exceed_length is not None else 0
        self.repetition_max_penalty = repetition_max_penalty
        self.repetition_ngram_size = repetition_ngram_size
        self.score_fn = score_fn

        if min_value_wrong > 0 or max_value_wrong > 0:
            raise ValueError("Wrong values should not be positive")

        logger.info(
            "Initialized math rule cosine scaled reward with"
            f" min_value_wrong: {min_value_wrong}, max_value_wrong: {max_value_wrong},"
            f" min_value_correct: {min_value_correct}, max_value_correct: {max_value_correct},"
            f" max_len: {max_len}, exceed_length: {exceed_length}, MAX_LEN_MARGIN: {MAX_LEN_MARGIN},"
            f" repetition_max_penalty: {repetition_max_penalty}, repetition_ngram_size: {repetition_ngram_size}")

    @staticmethod
    def is_selected(remote_rm_url: str) -> bool:
        return remote_rm_url == "math_rule_cosine_scale"

    def reward(
            self,
            sequences: List[str],
            gen_lengths: List[int],
            answers: List[str],
            scores: List[float],
            output_ids: List[List[int]],
            num_actions: int) -> List[float]:
        """Calculate correct/wrong rewards based solution length using a cosine schedule.

        The general idea is:
        - Shorter correct solutions should be rewarded over longer ones.
        - Longer wrong solutions should be rewarded over shorter ones.
        - Shorter solutions should be more risk averse (wrong penalized more than correct rewarded).
        """
        assert len(sequences) == len(gen_lengths)
        assert len(sequences) == len(answers)
        assert len(sequences) == len(scores)

        rewards = []
        for gen, gen_len, score in zip(sequences, gen_lengths, scores):
            if gen_len + MAX_LEN_MARGIN >= self.max_len:
                logger.info(f"Exceed length penalty applied -- gen_len: {gen_len}, max_len: {self.max_len}")
                rewards.append(self.exceed_length)
                continue

            if score == 1:
                min_value = self.min_value_correct
                max_value = self.max_value_correct
                rep_penalty = 0
            else:
                # Yes, they are swapped. This is required for the cosine formula below
                # to work with negative numbers.
                min_value = self.max_value_wrong
                max_value = self.min_value_wrong

                rep_penalty = get_repetition_penalty(
                    ngram_size=self.repetition_ngram_size,
                    max_penalty=self.repetition_max_penalty,
                    generation=gen)

            progress = gen_len / self.max_len
            cosine = math.cos(progress * math.pi)
            r = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            r += rep_penalty

            rewards.append(r)

        return rewards