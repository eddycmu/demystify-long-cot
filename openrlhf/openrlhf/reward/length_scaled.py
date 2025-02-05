import logging
from typing import List, Optional

from openrlhf.reward.common import MAX_LEN_MARGIN, SparseReward, score_hendrycks
from openrlhf.reward.repetition import get_repetition_penalty


logger = logging.getLogger(__name__)


class LengthScaledSparseReward(SparseReward):
    def __init__(self, risk_factor: float, max_len: int, exceed_length: Optional[float]):
        self.risk_factor = risk_factor
        self.max_len = max_len
        self.exceed_length = exceed_length if exceed_length is not None else 0
        logger.info(
            f"Initialized math rule length scaled reward with risk factor: {risk_factor},"
            f" max_len: {max_len}, exceed_length: {exceed_length}, MAX_LEN_MARGIN: {MAX_LEN_MARGIN}")

    @staticmethod
    def is_selected(remote_rm_url: str) -> bool:
        return remote_rm_url == "math_rule_len_scale"

    def reward(self, sequences: List[str], gen_lengths: List[int], answers: List[str], output_ids: List[List[int]], num_actions: int) -> List[float]:
        """Calculate correct/wrong rewards based solution length.

        The general idea is:
        - Shorter correct solutions should be rewarded over longer ones.
        - Longer wrong solutions should be rewarded over shorter ones.
        - Shorter solutions should be more risk averse (wrong penalized more than correct rewarded).
        - Risk should be equalized as the solution length increases.

        Formula:
        k: Risk factor -- the lower this is, the less we reward correct answers relative to penalizing wrong answers.
        n: Gen len
        n_max: Max len
        c: n/n_max
        correct reward: (k + (1 - k)c) * (1 - c)
        wrong reward: -1 + c
        """
        assert len(sequences) == len(gen_lengths)
        assert len(sequences) == len(answers)

        rewards = []
        for gen, gen_len, ans in zip(sequences, gen_lengths, answers):
            if gen_len + MAX_LEN_MARGIN >= self.max_len:
                logger.info(f"Exceed length penalty applied -- gen_len: {gen_len}, max_len: {self.max_len}")
                rewards.append(self.exceed_length)
                continue

            k = self.risk_factor
            c = gen_len / self.max_len
            if score_hendrycks(gen, ans) == 1:
                r = (k + (1 - k) * c) * (1 - c)
            else:
                r = -1 + c

            rewards.append(r)

        return rewards