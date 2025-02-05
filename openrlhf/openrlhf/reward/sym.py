import logging
from pebble.concurrent import process
from concurrent.futures import TimeoutError
from symeval import EvaluatorMath


logger = logging.getLogger(__name__)
TIMEOUT_SECS = 60


def score_symeval(predict: str, gold_raw: str) -> float:
    try:
        future = _score_symeval(predict, gold_raw)
        return future.result()
    except TimeoutError as e:
        logger.error(e)
        return 0
    except Exception:
        return 0


@process(timeout=TIMEOUT_SECS)
def _score_symeval(predict: str, gold_raw: str) -> float:
    evaluator = EvaluatorMath()
    gold = evaluator.extract_ans(gold_raw)
    resp = evaluator.extract_ans(predict)
    equal = evaluator.eq(gold, resp)
    return int(equal)
