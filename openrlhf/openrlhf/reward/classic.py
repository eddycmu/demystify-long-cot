import logging
from typing import Callable, List, Optional

from openrlhf.reward.common import MAX_LEN_MARGIN, SparseReward


logger = logging.getLogger(__name__)


class ClassicSparseReward(SparseReward):
    def __init__(
            self,
            correct: float,
            wrong: float,
            exceed_length: Optional[float],
            max_length: int,
            score_fn: Callable[[str, str], float]):
        self.correct = correct
        self.wrong = wrong
        self.exceed_length = exceed_length
        self.max_length = max_length
        self.score_fn = score_fn
        logger.info(
            f"Initialized math rule classic reward with correct: {correct},"
            f" wrong: {wrong}, exceed_length: {exceed_length}, max_length: {max_length}, MAX_LEN_MARGIN: {MAX_LEN_MARGIN}")

    @staticmethod
    def is_selected(remote_rm_url: str) -> bool:
        return remote_rm_url == "math_rule"

    def reward(self, sequences: List[str], gen_lengths: List[int], answers: List[str], scores: List[float], output_ids: List[List[int]], num_actions: int) -> List[float]:
        assert len(sequences) == len(gen_lengths)
        assert len(sequences) == len(answers)
        rewards = []
        for gen_len, sc in zip(gen_lengths, scores):
            if self.exceed_length is not None and gen_len + MAX_LEN_MARGIN >= self.max_length:
                logger.info(f"Exceed length penalty applied -- gen_len: {gen_len}, max_len: {self.max_length}")
                rewards.append(self.exceed_length)
                continue

            s = self.correct if sc == 1 else self.wrong
            rewards.append(s)
        return rewards
