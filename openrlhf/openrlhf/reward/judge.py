import re
from typing import List
import torch
import ray
from vllm import SamplingParams

TRUNCATED_LINES = 20
PROMPT_TEMPLATE = """Given the following last 20 lines of the LLM response to a math question
and the reference solution to that question, evaluate if the LLM response is correct based only on the LLM's final answer.

LLM response (last 20 lines):
...
{out}

Reference solution:
{ref}

Explain your thought process step-by-step before responding with `Judgement: <correct/wrong/not_found>`
"""


class Judge:
    def __init__(self, tokenizer, vllm_engines: List):
        assert tokenizer is not None
        self._tokenizer = tokenizer
        self._vllm_engines = vllm_engines

    def score(self, outputs: List[str], references: List[str]) -> List[float]:
        # round-robin load balance
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self._vllm_engines) <= world_size:
            llms = [self._vllm_engines[rank % len(self._vllm_engines)]]
        else:
            llms = self._vllm_engines[rank::world_size]

        sampling_params = SamplingParams(
            temperature=0,
            top_p=1,
            top_k=-1,
            max_tokens=1024,
            min_tokens=1,
            skip_special_tokens=False,
        )
        assert len(outputs) == len(references)
        all_prompts = []
        for out, ref in zip(outputs, references):
            out = "\n".join(out.split("\n")[-TRUNCATED_LINES:])
            prompt = [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(out=out, ref=ref),
                }
            ]
            all_prompts.append(prompt)

        all_prompt_token_ids = [
            self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True)
            for messages in all_prompts
        ]
        assert len(all_prompt_token_ids) == len(all_prompts)

        # Distribute requests to engines and collect responses to output
        all_judgement_refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            if prompt_token_ids:
                all_judgement_refs.append(
                    llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
                )

        # Retrieve and combine results from all outputs
        all_judgements = sum(ray.get(all_judgement_refs), [])

        scores = []
        for judgement in all_judgements:
            judgement_text = self._tokenizer.decode(judgement.outputs[0].token_ids, skip_special_tokens=True)
            sc = Judge._parse(judgement_text)
            scores.append(sc)

        assert len(scores) == len(outputs)
        return scores

    @staticmethod
    def _parse(text) -> int:
        pattern = r'.*Judgement:\s+(correct|wrong|not_found).*'
        match = re.search(pattern, text, re.IGNORECASE)
        extracted = match.group(1) if match else None
        return int(extracted == "correct")