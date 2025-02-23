# Demystifying Long Chain-of-Thought Reasoning in LLMs

This repo contains code used for experiments in the paper ["Demystifying Long Chain-of-Thought Reasoning in LLMs"](https://arxiv.org/abs/2502.03373). 


## News

- [2025/02/06] We released [the paper](https://arxiv.org/abs/2502.03373) and [the codebase](https://github.com/eddycmu/demystify-long-cot).

## TODOs

- [ ] Release action prompting code.
- [ ] Release run scripts for more sections.

## Introduction

Large language models have demonstrated remarkable reasoning abilities in domains like mathematics and programming.
A key technique for enabling reasoning abilities in LLMs is chain-of-thought (CoT) prompting, which guides models to generate intermediate reasoning steps before arriving at a final answer. Despite these advancements, LLMs still struggle with highly complex reasoning tasks, such as mathematical competitions.

Recently, OpenAI’s o1 and DeepSeek R1 models have demonstrated significant breakthroughs in these tasks. A key distinguishing feature of these models is their ability to scale up inference compute with long CoTs, which include strategies such as recognizing and correcting mistakes, breaking down difficult steps, and iterating on alternative approaches, leading to substantially longer and more structured reasoning processes.

Several efforts have attempted to replicate the performance of o1 models by training LLMs to generate long CoTs. 
However, a comprehensive understanding of how models learn and generate long CoTs remains limited. In this work, we systematically investigate the underlying mechanics of long CoT generation.

In order to run the experiments that resulted in the findings of the paper, we implemented the following main changes in our fork of OpenRLHF:

1. Support for rule-based reward functions via the remote reward model code path.
2. Different rule-based reward functions, including the Cosine Reward that stabilizes and controls CoT length using incentives.
3. Support for multiple reward types having different discount factors (gamma) for both PPO (via GAE) and Reinforce++.
4. LLM-as-a-judge as a reference-guided verifier, which can be used as an alternative to rule-based verification.
It is compatible with the rule-based reward functions mentioned above.

We also included the  `minhash` code we used to search through pre-training data for reasoning patterns characteristic of long CoT.

## Quick Start

Firstly, set up your OpenRLHF environment the [usual way](https://github.com/OpenRLHF/OpenRLHF?tab=readme-ov-file#quick-start) as instructed below (from the OpenRLHF docs). Use our [dependency installation script](openrlhf/scripts/install_deps.sh) to install additional dependencies, but depending on your environment, some details may vary.

Secondly, you can consider using one of our experiments [run scripts](openrlhf/scripts) as a starting point for your own exploration. Note that file paths and api keys were removed with search-and-replace, so minor fixes might be required before the scripts are runnable.

### Installation

To use OpenRLHF, first launch the docker container (**Recommended**) and `pip install` openrlhf inside the docker container:

```bash
# Launch the docker container
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:24.07-py3 bash
sudo pip uninstall xgboost transformer_engine flash_attn -y

# pip install
pip install openrlhf

# If you want to use vLLM acceleration (Install vLLM 0.6.5)
pip install openrlhf[vllm]
# latest vLLM is also supported
pip install openrlhf[vllm_latest]

# pip install the latest version
pip install git+https://github.com/OpenRLHF/OpenRLHF.git

# Or git clone
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF
pip install -e .
```

> [!NOTE]
>We recommend using vLLM 0.6.4 or higher. Other versions (vLLM >= 0.4.2) may require weight synchronization via Gloo (`--vllm_sync_backend gloo`).
>We also provided the [Dockerfiles for vLLM](./dockerfile/) and [One-Click Installation Script of Nvidia-Docker](./examples/scripts/nvidia_docker_install.sh).

## Codebase Pointers

- [Run script directory](openrlhf/scripts/)
- [Reward function implementations](openrlhf/openrlhf/reward)
- [GAE: Multiple Reward Types](openrlhf/openrlhf/trainer/ppo_utils/experience_maker.py#L421)
- [LLM-as-a-judge](openrlhf/openrlhf/reward/judge.py)
- [MinHash for searching through Pre-training Data](minhash/)

## Acknowledgements

We would like to thank those who have contributed to the following projects, upon which we have built our research: OpenRLHF, vLLM, DeepSpeed, Llama, Qwen, DeepSeek R1, Kimi-k1.5 and o1. We are also grateful to many others who have helped but are not mentioned here.

## Citation

If you find our work helpful, please kindly cite us as:

```bibtex
@misc{yeotong2025longcot,
      title={Demystifying Long Chain-of-Thought Reasoning in LLMs}, 
      author={Edward Yeo and Yuxuan Tong and Morry Niu and Graham Neubig and Xiang Yue},
      year={2025},
      eprint={2502.03373},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.03373}, 
}
```
