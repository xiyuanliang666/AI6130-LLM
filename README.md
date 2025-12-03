# AI6130-LLM
This repository contains the implementation and experimental results of two core assignments for the AI6130 course, focusing on **LLM benchmark evaluation with prompt engineering** and **parameter-efficient fine-tuning (PEFT) for mathematical reasoning**.

## Task Requirement
### Assignment 1: Benchmark Evaluation with Open-LLM
1. Load three tasks from Open-LLM-Benchmark: CommonsenseQA, OpenbookQA, PiQA.
2. Load four LLMs: TinyLlama_v1.1, Qwen/Qwen2.5-3B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B, Qwen/Qwen3-4B-Instruct-2507, and run each model on the three tasks to record evaluation results.
3. Modify the prompt in the infer_llm function (e.g., specify answer format, add few-shot examples) while ensuring compatibility with model templates, then re-run all models on the three tasks and record new results.
### Assignment 2: Parameter-Efficient Fine-Tuning of Large Language Models
1. Load two LLMs: OpenLLaMA (openlm-research/open_llama_3b_v2), TinyLlama_v1.1.
2. Use the math_7k.json dataset and adopt LORA, AdapterP methods for model fine-tuning.
3. Evaluate the fine-tuned models on two benchmarks: AQuA, AddSub.
4. (Optional) Conduct additional experiments with different settings, such as adjusting the number of epochs, using other datasets (e.g., math_10k.json), testing more LLMs, or adding evaluation benchmarks (MultiArith, SingleEq, gsm8k, SVAMP).


## Assignment 1: LLM Benchmark Evaluation with Prompt Engineering
### Core Work
1. Conducted baseline evaluation of 4 LLMs (TinyLlama_v1.1, Qwen2.5-3B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-4B-Instruct-2507) on 3 benchmarks (CommonsenseQA, OpenbookQA, PiQA).
2. Performed prompt optimization: added task-specific format constraints, adapted prompts to model templates (e.g., ChatML for Qwen series), adjusted parameters (max_new_tokens, do_sample) and added regex extraction for redundant output filtering.
3. Implemented few-shot prompting with task-matched examples to further improve model performance.
### Key Results
- Baseline accuracy was extremely low (0%-6.67%) due to unconstrained prompts and irregular outputs.
- After prompt optimization, Qwen3-4B-Instruct-2507 achieved excellent performance with 79.86% on CommonsenseQA, 83.10% on OpenbookQA, and 85.20% on PiQA.
- Few-shot prompting boosted TinyLlama_v1.1's accuracy by 8%-10%, while medium and large models had minimal improvement, with Qwen2.5-3B-Instruct even seeing a 4.68% drop on OpenbookQA.
- The final top model Qwen3-4B-Instruct-2507 reached 80.56% on CommonsenseQA, 83.3% on OpenbookQA, and 87.36% on PiQA after comprehensive optimization.
### Analysis & Findings
- Model performance is determined by size, instruction tuning, and distillation: larger instruction-tuned models (Qwen3 > Qwen2.5) outperform smaller or distilled models, and DeepSeek-R1's - math-optimized base model is ill-suited for commonsense tasks.
- Task-specific prompt constraints and model template compatibility are critical to improving output quality, resolving issues like unstructured responses and non-compliant formats.
- Few-shot prompting is more effective for weak models (TinyLlama_v1.1) as it provides task paradigms, while strong models have limited room for improvement and may be misled by a small number of examples.

## Assignment 2: Parameter-Efficient Fine-Tuning of LLMs
### Core Work
1. Carried out base experiments: fine-tuned 2 LLMs (OpenLLaMA-3B-v2, TinyLlama-v1.1) with 2 PEFT methods (LoRA, AdapterP) on the math_7k.json dataset, and evaluated on AQuA and AddSub benchmarks.
2. Conducted extension experiments: tested the impact of different epochs, additional evaluation benchmarks (SingleEq, SVAMP, MultiArith, GSM8K), alternative datasets (math_10k.json) and another LLM (llama-7b-hf) on fine-tuning results.
### Key Results
- Base experiments: TinyLlama-v1.1 + LoRA achieved the highest AQuA accuracy (22.44%), and OpenLLaMA-3B-v2 + LoRA led in AddSub accuracy (39.75%).
- Epoch extension: Increasing epochs brought limited improvement, with AQuA accuracy dropping to 16.54% at epoch 2 and AddSub rising to 42.28%.
- Additional benchmarks: MultiArith achieved the highest accuracy (67%), while GSM8K had the lowest (18.27%).
- Dataset switch: math_10k.json boosted AddSub accuracy to 67.34% but reduced AQuA accuracy to 16.93%.
- Other LLM test: llama-7B-HF underperformed OpenLLaMA-3B-v2, with 13.78% on AQuA and 33.16% on AddSub.
### Analysis & Findings
- PEFT method adaptability varies by task: LoRA excels at numerical recognition tasks (AddSub) and small models, while AdapterP is slightly better for multi-step reasoning tasks (AQuA).
- Model performance depends more on pretraining quality than size: OpenLLaMA-3B-v2's math-focused pretraining corpus makes it outperform the larger llama-7B-HF.
- Task structure affects accuracy: Structured, pattern-regular tasks (MultiArith, SingleEq) yield higher results, while linguistic perturbations (SVAMP) and unstructured outputs reduce performance.
- Training data distribution is crucial: math_7k.json (reasoning-focused) aligns better with AQuA, and overexposure to arithmetic data in math_10k.json weakens reasoning ability.

