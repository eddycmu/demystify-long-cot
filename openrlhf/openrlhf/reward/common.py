from abc import ABC, abstractmethod
import re
from typing import List
import torch


# We add this to the gen len before checking if it is above max len to ensure there are no
# off-by-one mistakes.
MAX_LEN_MARGIN = 16


def is_math_rule_reward(remote_rm_url: str) -> bool:
    return remote_rm_url.startswith("math_rule")


class SparseReward(ABC):
    @staticmethod
    @abstractmethod
    def is_selected(remote_rm_url: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def reward(self, *args, **kwargs) -> List[float]:
        raise NotImplementedError


class DenseReward(ABC):
    @staticmethod
    @abstractmethod
    def is_selected(remote_rm_url: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def reward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


def score_hendrycks(predict: str, gold_raw: str) -> float:
    gold = remove_boxed(last_boxed_only_string(gold_raw))
    if gold is None:
        gold = gold_raw

    answer = postprocess(predict)
    # Find 'The answer is' as we specified in fewshot.
    extracted_answer = answer
    if "answer is" in predict.lower():
        idx = predict.lower().rfind("answer is") + len("answer is")
        extracted_answer = postprocess(predict.lower()[idx:])
    # extract content within $$ and []
    before_predict = "\\" + predict[:-1]
    post_predict = predict[1:] + "\\"
    indices = [
        pos
        for pos, (char, before_char, post_char) in enumerate(zip(predict, before_predict, post_predict))
        if (char == "$" and (before_char != "\\" or pos == 0)) or (char == "\\" and post_char == "]")
    ]
    # find any possible answer in the predict string, should be the uppper bound of acc
    any_acc = 0.0
    if len(indices) <= 1:
        answer = predict
        # find the final answer in the string ex. $36-26=10$, find 10
        if answer.rfind("=") >= 0:
            answer = answer[answer.rfind("=") + 1 :]
        answer_rmboxed = remove_boxed(last_boxed_only_string(answer))
        any_acc = float(is_equiv(answer, gold))
        any_acc = max(any_acc, float(is_equiv(answer_rmboxed, gold)))
    else:
        # find every possible answer wrap in LaTeX fomula
        for i in range(len(indices) - 1):
            answer = predict[indices[i] + 1 : indices[i + 1]]
            # find the final answer in the string ex. $36-26=10$, find 10
            if answer.rfind("=") >= 0:
                answer = answer[answer.rfind("=") + 1 :]
            answer_rmboxed = remove_boxed(last_boxed_only_string(answer))
            _ans_acc = float(is_equiv(answer, gold))
            _ans_rmboxed_acc = float(is_equiv(answer_rmboxed, gold))
            any_acc = max(any_acc, _ans_acc, _ans_rmboxed_acc)
    answer_rmboxed = remove_boxed(last_boxed_only_string(answer))
    answer_extract = remove_boxed(last_boxed_only_string(extracted_answer))
    match = re.search(r"\\boxed\{(.+)\}", predict)
    answer_re = match.group(1) if match is not None else ""

    try:
        meta_math_answer = meta_math_extract_answer(predict)
    except Exception:
        meta_math_answer = "ANSWER_NOT_FOUND"

    acc = float(
        max(
            is_equiv(answer, gold),
            is_equiv(answer_rmboxed, gold),
            is_equiv(answer_extract, gold),
            is_equiv(answer_re, gold),
            is_equiv(meta_math_answer, gold),
        )
    )
    return acc


def sparse_to_dense(action_mask: torch.Tensor, reward: List[float]) -> torch.Tensor:
    r = torch.tensor(reward, device=action_mask.device)
    eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
    return torch.zeros_like(action_mask, dtype=r.dtype).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1))


def sparse_to_dense_gen_length(batch_size: int, num_actions: int, gen_lengths: List[int], reward: List[float]) -> torch.Tensor:
    r = torch.tensor(reward)
    eos_indices = (torch.tensor(gen_lengths, dtype=torch.long) - 1).unsqueeze(1)
    return torch.zeros(batch_size, num_actions, dtype=r.dtype).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1))


def last_boxed_only_string(string: str) -> str:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return s


# From opencompass https://github.com/InternLM/opencompass/blob/main/opencompass/datasets/math.py
def postprocess(text: str) -> str:
    SUBSTITUTIONS = [
        ("an ", ""),
        ("a ", ""),
        (".$", "$"),
        ("\\$", ""),
        ("$$", "$"),  # Let consecutive dollar sign to be single
        (r"\ ", ""),
        ("mbox", "text"),
        (",\\text{and}", ","),
        ("\\text{and}", ","),
        ("\\text{m}", "\\text{}"),
        ("\\le", "<"),
    ]
    REMOVED_EXPRESSIONS = [
        "square",
        "ways",
        "integers",
        "dollars",
        "mph",
        "inches",
        "ft",
        "hours",
        "km",
        "units",
        "\\ldots",
        "sue",
        "points",
        "feet",
        "minutes",
        "digits",
        "cents",
        "degrees",
        "cm",
        "gm",
        "pounds",
        "meters",
        "meals",
        "edges",
        "students",
        "childrentickets",
        "multiples",
        "\\text{s}",
        "\\text{.}",
        "\\text{\ns}",
        "\\text{}^2",
        "\\text{}^3",
        "\\text{\n}",
        "\\text{}",
        r"\mathrm{th}",
        r"^\circ",
        r"^{\circ}",
        r"\;",
        r",\!",
        "{,}",
        '"',
        "\\dots",
        "\n",
        "\r",
        "\f",
    ]

    def normalize_final_answer(final_answer: str) -> str:
        """Normalize a final answer to a quantitative reasoning question."""
        # final_answer = final_answer.split('=')[-1]
        for before, after in SUBSTITUTIONS:
            final_answer = final_answer.replace(before, after)
        for expr in REMOVED_EXPRESSIONS:
            final_answer = final_answer.replace(expr, "")

        # Extract answer that is in LaTeX math, is bold,
        # is surrounded by a box, etc.
        final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
        assert "\n" not in final_answer
        assert "\r" not in final_answer
        assert "\f" not in final_answer
        if len(re.findall(r"finalansweris(.*)", final_answer)) > 0:
            final_answer = re.findall(r"finalansweris(.*)", final_answer)[-1]

        if len(re.findall(r"oxed\{(.*?)\}", final_answer)) > 0:
            final_answer = re.findall(r"oxed\{(.*?)\}", final_answer)[-1]

        if len(re.findall(r"\$(.*?)\$", final_answer)) > 0:
            final_answer = re.findall(r"\$(.*?)\$", final_answer)[-1]
        final_answer = final_answer.strip()
        if "rac" in final_answer and "\\frac" not in final_answer:
            final_answer = final_answer.replace("rac", "\\frac")
        # Normalize shorthand TeX:
        # \fracab -> \frac{a}{b}
        # \frac{abc}{bef} -> \frac{abc}{bef}
        # \fracabc -> \frac{a}{b}c
        # \sqrta -> \sqrt{a}
        # \sqrtab -> sqrt{a}b
        final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
        final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
        final_answer = final_answer.replace("$", "")

        # Normalize 100,000 -> 100000
        if final_answer.replace(",", "").isdigit():
            final_answer = final_answer.replace(",", "")

        return final_answer

    for maybe_ans in text.split("."):
        if "targets" in maybe_ans.lower():
            return normalize_final_answer(maybe_ans)
    return normalize_final_answer(text.split(".")[0])


def remove_brace(text):
    pattern = r"^\((.*?)\)$"
    match = re.search(pattern, text)
    if match:
        extracted_text = match.group(1)  # Access the matched content
        return extracted_text
    return text


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # Normalize 100,000 -> 100000
    if string.replace(",", "").isdigit():
        string = string.replace(",", "")
    string = remove_brace(string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2},
    # etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def meta_math_extract_answer(answer: str) -> str:
    ans_prefix = "the answer is:"
    extracted = answer[answer.lower().index(ans_prefix) + len(ans_prefix) :]
    return extract(extracted)


def extract(answer: str) -> str:
    extracted = answer.strip()

    if extracted[-1] == ".":
        extracted = extracted[:-1]

    if extracted[:2] == r"\(" and extracted[-2:] == r"\)":
        extracted = extracted[2:-2]

    if extracted[:2] == "$$" and extracted[-2:] == r"$$":
        extracted = extracted[2:-2]

    if extracted[:2] == r"(" and extracted[-2:] == r")":
        extracted = extracted[2:-2]

    if extracted[0] == "$":
        extracted = extracted[1:]

    if extracted[-1] == "$":
        extracted = extracted[:-1]

    extracted = extracted.strip()
    return extracted