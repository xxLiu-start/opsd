# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import re
import textwrap
import warnings
import json
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import DistributedType, broadcast_object_list, gather_object, is_peft_model
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.integration_utils import is_wandb_available
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction
from transformers.utils import (
    is_flash_attn_2_available,
    is_liger_kernel_available,
    is_peft_available,
    is_rich_available,
)

from trl.data_utils import is_conversational, maybe_convert_to_chatml, pack_dataset, truncate_dataset
from trl.extras.profiling import profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_vllm_available
from trl.models import prepare_deepspeed
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.utils import (
    DataCollatorForChatML,
    disable_dropout_in_model,
    empty_cache,
    ensure_master_addr_port,
    pad,
)
from trl.experimental.gold.gold_config import GOLDConfig
from data_collator import SelfDistillationDataCollator

try:
    from math_verify import parse, verify
except ImportError:
    parse = verify = None


if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text


def _extract_boxed_answer(text: str | None) -> str | None:
    if not text:
        return None

    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>") :] if think_end != -1 else text

    idx = search_text.find(r"\boxed{")
    if idx == -1:
        return None

    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(search_text) and depth > 0:
        if search_text[i] == "{":
            depth += 1
        elif search_text[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        return search_text[start : i - 1].strip()
    return None


def _preprocess_for_parse(answer: str | None) -> str | None:
    if answer is None:
        return None
    ratio_match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*", answer)
    if ratio_match:
        return rf"\frac{{{ratio_match.group(1)}}}{{{ratio_match.group(2)}}}"
    return answer


def reward_correctness_from_solutions(completions: list[str], solutions: list[str]) -> list[float]:
    rewards = []
    for completion, solution in zip(completions, solutions):
        pred_answer = _extract_boxed_answer(completion)
        gt_answer = _extract_boxed_answer(solution) or (solution.strip() if solution else "")
        reward = 0.0

        if parse is not None and verify is not None:
            gold_parsed = parse(gt_answer)
            pred_parsed = parse(_preprocess_for_parse(pred_answer))
            if gold_parsed is not None and pred_parsed is not None:
                try:
                    reward = 1.0 if verify(gold_parsed, pred_parsed) else 0.0
                except Exception:
                    reward = 0.0

        if reward == 0.0:
            pred_norm = re.sub(r"\s+", "", pred_answer or "").lower()
            gt_norm = re.sub(r"\s+", "", gt_answer or "").lower()
            if pred_norm and pred_norm == gt_norm:
                reward = 1.0

        rewards.append(reward)

    return rewards


class EMAUpdateCallback(TrainerCallback):
    """Update EMA teacher weights after each optimizer step."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Only update when the optimizer actually stepped (end of a gradient accumulation cycle)
        if self.trainer.use_ema_teacher and self.trainer.accelerator.sync_gradients:
            self.trainer._update_ema()


class GOLDVLLMSyncCallback(TrainerCallback):
    """Sync the model weights to vLLM after training steps when it's safe to do so."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Sync weights after training step when DeepSpeed is stable."""
        if (
            self.trainer.use_vllm
            and state.global_step != self.trainer._last_vllm_sync_step
            and state.global_step % self.trainer.vllm_sync_frequency == 0
        ):
            # Check if this is a step where gradients are synchronized
            # This happens at the end of gradient accumulation cycles
            if (
                hasattr(self.trainer.accelerator, "sync_gradients")
                and self.trainer.accelerator.sync_gradients
            ):
                self.trainer._move_model_to_vllm()
                self.trainer._last_vllm_sync_step = state.global_step


class PeriodicSFTCallback(TrainerCallback):
    """Trigger remediation SFT after optimizer steps."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.trainer.accelerator.sync_gradients:
            self.trainer._maybe_run_periodic_sft()


class OPSDTrainer(SFTTrainer):
    _tag_names = ["trl", "opsd"]
    _name = "OPSD"

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        args: GOLDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: (
            PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None
        ) = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Optional["PeftConfig"] = None,
        use_thinking_machines_loss: bool = False,
        fixed_teacher: bool = False,
        reason_first: bool = False,
        top_k_loss: int | None = None,
        use_ema_teacher: bool = False,
        ema_decay: float = 0.999,
        num_generations: int = 1,
        low_advantage_threshold: float = -0.5,
        periodic_sft_interval: int = 1000,
        periodic_sft_dataset_size: int = 5000,
        periodic_sft_epochs: int = 1,
        periodic_sft_batch_size: int = 1,
        periodic_sft_learning_rate: float = 5e-6,
        periodic_sft_max_samples: int = 0,
    ):
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path
        self.model_revision = getattr(args, "student_model_revision", None)
        if isinstance(model, str) and self.model_revision is not None:
            args.model_init_kwargs = args.model_init_kwargs or {}
            args.model_init_kwargs.setdefault("revision", self.model_revision)

        # Custom data collator for self-distillation
        if data_collator is None:
            data_collator = SelfDistillationDataCollator(
                tokenizer=processing_class, max_length=args.max_length, reason_first=reason_first
            )

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.seq_kd = args.seq_kd
        self.use_thinking_machines_loss = use_thinking_machines_loss
        self.fixed_teacher = fixed_teacher
        self.reason_first = reason_first
        self.top_k_loss = top_k_loss
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
        self.num_generations = max(1, int(num_generations))
        self.low_advantage_threshold = low_advantage_threshold
        self.periodic_sft_interval = periodic_sft_interval
        self.periodic_sft_dataset_size = periodic_sft_dataset_size
        self.periodic_sft_epochs = periodic_sft_epochs
        self.periodic_sft_batch_size = periodic_sft_batch_size
        self.periodic_sft_learning_rate = periodic_sft_learning_rate
        self.periodic_sft_max_samples = periodic_sft_max_samples
        self._ema_params = None  # lazily initialized on first optimizer step
        self._last_low_advantage_flags = None
        self._last_response_advantages = None
        self._sft_running = False

        remediation_dir = Path(self.args.output_dir) / "remediation"
        remediation_dir.mkdir(parents=True, exist_ok=True)
        self.remediation_dir = remediation_dir
        self.low_adv_student_path = remediation_dir / "low_adv_student_responses.jsonl"
        self.low_adv_teacher_path = remediation_dir / "low_adv_teacher_responses.jsonl"
        self.pending_sft_path = remediation_dir / "pending_sft_examples.jsonl"
        self.sft_windows_dir = remediation_dir / "sft_windows"
        self.sft_windows_dir.mkdir(parents=True, exist_ok=True)
        self.sft_checkpoint_dir = remediation_dir / "sft_checkpoints"
        self.sft_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.generations_dir = Path(self.args.output_dir) / "generations"
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        self.question_response_dir = self.generations_dir / "question_rollouts"
        self.question_response_dir.mkdir(parents=True, exist_ok=True)
        self.sample_text_log_path = self.generations_dir / "sample_text.log"

        # Validate fixed_teacher option
        if self.fixed_teacher and peft_config is None:
            raise ValueError(
                "fixed_teacher=True requires a PEFT config (use_peft=True). "
                "The fixed teacher is implemented by disabling LoRA adapters during teacher forward passes."
            )

        if self.use_ema_teacher and self.fixed_teacher:
            raise ValueError(
                "use_ema_teacher=True and fixed_teacher=True are mutually exclusive teacher strategies."
            )

        if self.use_ema_teacher:
            self.add_callback(EMAUpdateCallback(self))
            print(f"\n{'='*80}")
            print("EMA TEACHER MODE ENABLED")
            print(f"EMA decay: {self.ema_decay}")
            print("Teacher is an exponential moving average of the student weights.")
            print("EMA parameters are initialized on the first optimizer step.")
            print(f"{'='*80}\n")

        if self.fixed_teacher:
            print(f"\n{'='*80}")
            print("FIXED TEACHER MODE ENABLED")
            print("Teacher will use the initial policy (base model without LoRA adapters)")
            print("Student will update with LoRA adapters")
            print(f"{'='*80}\n")

        if self.reason_first:
            print(f"\n{'='*80}")
            print("REASON FIRST MODE ENABLED")
            print("Teacher will first reason about the privileged solution, then evaluate student's response")
            print(f"{'='*80}\n")

        if self.periodic_sft_dataset_size > 0:
            self.add_callback(PeriodicSFTCallback(self))

        # Track per-step loss statistics for on/off-policy batches (used in logging)
        self._on_policy_loss_total = 0.0
        self._off_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._off_policy_step_equiv = 0.0

        self.use_transformers_paged = args.use_transformers_paged or False

        # Track generation outputs for saving
        self._generation_outputs_buffer = []
        self._generation_save_frequency = 100
        self._window_save_frequency = 100
        self._question_response_buffer = []
        self._sft_window_buffer = []

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Generation config for reasoning phase (when reason_first=True)
        max_reasoning_length = getattr(args, "max_reasoning_length", 4096)
        self.reasoning_generation_config = GenerationConfig(
            max_new_tokens=max_reasoning_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.reasoning_generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.log_completion_steps = args.log_completions_steps
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.steps_per_generation
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
            "advantages": deque(maxlen=maxlen),
        }

        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.vllm_mode = args.vllm_mode
            self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
            self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
            self.vllm_enable_sleep_mode = args.vllm_enable_sleep_mode
            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    self.vllm_client = VLLMClient(
                        host=args.vllm_server_host,
                        server_port=args.vllm_server_port,
                        connection_timeout=args.vllm_server_timeout,
                    )
                    self.vllm_client.init_communicator()
            elif self.vllm_mode == "colocate":
                student_model_name_or_path = self.model_name_or_path

                # Make sure tensor_parallel_size divides world size evenly
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP
                    self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(
                                range(
                                    i * self.vllm_tensor_parallel_size,
                                    (i + 1) * self.vllm_tensor_parallel_size,
                                )
                            )
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                ensure_master_addr_port()

                self.vllm_engine = LLM(
                    model=student_model_name_or_path,
                    revision=self.model_revision,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.args.gradient_accumulation_steps,
                    max_model_len=args.max_length,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    enable_sleep_mode=self.vllm_enable_sleep_mode,
                )

                if self.vllm_enable_sleep_mode:
                    self.vllm_engine.sleep(level=2)

                # When using vLLM, the main process is responsible for loading the model weights. This can cause process
                # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
                # synchronize all processes after vLLM has been fully initialized.
                self.accelerator.wait_for_everyone()
            else:
                raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")
            self.vllm_guided_decoding_regex = args.vllm_guided_decoding_regex
            self.vllm_sync_frequency = args.vllm_sync_frequency
            self._last_vllm_sync_step = -1

            self.add_callback(GOLDVLLMSyncCallback(self))

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        required_columns = [
            "problem",
            "solution",
        ]
        if self._signature_columns is None:
            self._signature_columns = required_columns
        else:
            for column in required_columns:
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    # Knowledge distillation loss
    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        logits_are_probs=False,
        top_k=None,
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature:
                Softmax temperature (default: 1.0)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')
            top_k:
                If set, restricts the loss to only the top-k tokens of the teacher distribution. Both student and
                teacher distributions are renormalized over these k tokens before computing JSD. This reduces memory
                and focuses distillation on the teacher's most probable tokens. (default: None = full vocabulary)

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        if logits_are_probs:
            student_log_probs = torch.log(student_logits.clamp_min(1e-8))
            teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
        else:
            # Apply temperature scaling to logits before computing probabilities
            student_logits = student_logits / temperature
            teacher_logits = teacher_logits / temperature

            if top_k is not None and top_k > 0:
                # Restrict to top-k tokens of the teacher distribution and renormalize.
                # Shape: [batch, seq_len, top_k]
                _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
                student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
                teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

            # Compute log probabilities for student and probabilities for teacher
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    @staticmethod
    def generalized_jsd_loss_rl(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        logits_are_probs=False,
        top_k=None,
    ):
        """
        Compute a policy-gradient surrogate whose gradient matches generalized_jsd_loss.

        The sampled tokens in labels are treated as on-policy actions from the student. For 0 < beta < 1, the
        generalized JSD gradient with respect to the student distribution is:

            grad JSD = E_{a ~ pi_student}[
                (1 - beta) * (log pi_student(a) - log pi_mix(a)) * grad log pi_student(a)
            ]

        where pi_mix = (1 - beta) * pi_student + beta * pi_teacher. This is equivalent to the RL loss:

            reward = (1 - beta) * (log pi_mix(a) - log pi_student(a))
            loss   = - stop_grad(reward) * log pi_student(a)

        The beta=1 endpoint matches KL(student || teacher). The beta=0 endpoint matches KL(teacher || student)
        using an on-policy importance-weighted estimator.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1, matching generalized_jsd_loss.
            temperature:
                Temperature applied before computing action log-probs (default: 1.0)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')
            top_k:
                Kept for backward-compatible call sites. Not used because the sampled action must remain available.

        Returns:
            loss: Scalar tensor with the RL-style policy objective
        """
        del logits_are_probs, top_k

        if labels is None:
            raise ValueError("labels are required for the RL-style objective because they identify sampled actions.")

        if beta < 0 or beta > 1:
            raise ValueError(f"beta must be between 0 and 1, got {beta}.")

        mask = labels != -100
        safe_labels = labels.masked_fill(~mask, 0)

        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        student_log_probs_sampled = torch.gather(
            student_log_probs, dim=-1, index=safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        teacher_log_probs_sampled = torch.gather(
            teacher_log_probs, dim=-1, index=safe_labels.unsqueeze(-1)
        ).squeeze(-1)

        if beta == 0:
            # generalized_jsd_loss(beta=0) is KL(teacher || student). With actions sampled from the student,
            # q(a) / p(a) is the importance weight for an unbiased score-function estimator.
            reward = torch.exp(teacher_log_probs_sampled - student_log_probs_sampled)
        elif beta == 1:
            # generalized_jsd_loss(beta=1) is KL(student || teacher).
            reward = teacher_log_probs_sampled - student_log_probs_sampled
        else:
            beta_tensor = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs_sampled = torch.logsumexp(
                torch.stack(
                    [
                        student_log_probs_sampled + torch.log1p(-beta_tensor),
                        teacher_log_probs_sampled + torch.log(beta_tensor),
                    ]
                ),
                dim=0,
            )
            reward = (1 - beta_tensor) * (mixture_log_probs_sampled - student_log_probs_sampled)

        loss = -(reward.detach() * student_log_probs_sampled)
        loss = loss[mask]

        # Apply reduction
        if reduction == "batchmean":
            return loss.sum() / mask.sum().clamp_min(1)
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "mean":
            return loss.mean()
        else:
            return loss

    
    def _update_ema(self):
        """Update EMA parameters after an optimizer step.

        On the very first call this lazily initializes the EMA state as an exact copy of the
        current (trainable) model parameters, then returns without applying a decay step.
        Subsequent calls apply: ema = decay * ema + (1 - decay) * student.

        Only trainable parameters are tracked (i.e. LoRA adapter weights for PEFT models,
        or all parameters for full fine-tuning).

        ZeRO-3 note: with ZeRO-3 each rank only holds a shard of every parameter.
        We use `deepspeed.zero.GatheredParameters` (read-only, modifier_rank=None) so that
        every rank sees the full parameter tensor when snapshotting / updating the EMA.
        The EMA tensors are therefore full-sized copies, which is also required by
        `_ema_teacher_context` when it swaps the gathered student weights with EMA values.
        """
        decay = self.ema_decay
        unwrapped = self.accelerator.unwrap_model(self.model)

        # Detect ZeRO-3 (same pattern used elsewhere in this file)
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            trainable = [(name, param) for name, param in unwrapped.named_parameters() if param.requires_grad]
            params_list = [p for _, p in trainable]

            # modifier_rank=None → read-only gather; original partitions are restored on exit.
            with deepspeed.zero.GatheredParameters(params_list):
                if self._ema_params is None:
                    self._ema_params = {name: param.data.clone().detach() for name, param in trainable}
                    n_tensors = len(self._ema_params)
                    n_params = sum(p.numel() for p in self._ema_params.values())
                    print(
                        f"\nEMA teacher initialized: {n_tensors} tensors, {n_params:,} parameters "
                        f"(decay={decay})"
                    )
                    return  # first call = initialization only, no decay update

                for name, param in trainable:
                    if name not in self._ema_params:
                        continue
                    ema = self._ema_params[name]
                    if ema.device != param.data.device:
                        ema = ema.to(param.data.device)
                        self._ema_params[name] = ema
                    ema.mul_(decay).add_(param.data, alpha=1.0 - decay)
        else:
            if self._ema_params is None:
                # Lazy init: snapshot the current weights as the initial EMA state.
                self._ema_params = {
                    name: param.data.clone().detach()
                    for name, param in unwrapped.named_parameters()
                    if param.requires_grad
                }
                n_tensors = len(self._ema_params)
                n_params = sum(p.numel() for p in self._ema_params.values())
                print(
                    f"\nEMA teacher initialized: {n_tensors} tensors, {n_params:,} parameters "
                    f"(decay={decay})"
                )
                return  # first call = initialization only, no decay update

            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                # Move EMA buffer to the same device as the live param (handles multi-GPU setups)
                if ema.device != param.data.device:
                    ema = ema.to(param.data.device)
                    self._ema_params[name] = ema
                ema.mul_(decay).add_(param.data, alpha=1.0 - decay)

    @contextmanager
    def _ema_teacher_context(self, model):
        """Context manager that temporarily loads EMA weights for the teacher forward pass.

        Swaps `param.data` of every tracked (trainable) parameter with its EMA counterpart,
        runs the body (teacher forward), then restores the student weights unconditionally.
        Safe to use inside `torch.no_grad()`.  If EMA has not been initialized yet (step 0),
        this is a no-op and the current student weights are used instead.

        ZeRO-3 note: direct `param.data` assignment bypasses ZeRO-3's shard lifecycle and
        corrupts its internal state, causing size-mismatch errors during gradient-checkpoint
        recomputation.  When ZeRO-3 is active we therefore wrap the swap inside
        `deepspeed.zero.GatheredParameters` so the parameters are fully materialised on every
        rank before we touch them, and ZeRO-3 re-partitions cleanly when the context exits.
        """
        if self._ema_params is None:
            yield  # EMA not yet initialized; fall back to current weights
            return

        unwrapped = self.accelerator.unwrap_model(model)

        # Detect ZeRO-3 (same pattern used elsewhere in this file)
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            name_to_param = {
                name: param
                for name, param in unwrapped.named_parameters()
                if param.requires_grad and name in self._ema_params
            }
            params_list = list(name_to_param.values())

            # modifier_rank=0 causes ZeRO-3 to re-partition from rank-0's param.data on exit,
            # which will be the restored student weights.
            with deepspeed.zero.GatheredParameters(params_list, modifier_rank=0):
                saved = {}
                for name, param in name_to_param.items():
                    ema = self._ema_params[name]
                    if ema.device != param.data.device:
                        ema = ema.to(param.data.device)
                        self._ema_params[name] = ema
                    saved[name] = param.data.clone()
                    param.data.copy_(ema)
                try:
                    yield
                finally:
                    for name, param in name_to_param.items():
                        if name in saved:
                            param.data.copy_(saved[name])
        else:
            saved = {}
            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                if ema.device != param.data.device:
                    ema = ema.to(param.data.device)
                    self._ema_params[name] = ema
                saved[name] = param.data
                param.data = ema
            try:
                yield
            finally:
                for name, param in unwrapped.named_parameters():
                    if name in saved:
                        param.data = saved[name]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the self-distillation loss with an RLSD-style advantage flow.

        The loss is built from sampled token log-probabilities:
          advantage = log π_teacher(a|x) - log π_student(a|x)
          loss      = -E[stop_grad(advantage) * log π_student(a|x)]

        This keeps the model and dataset usage unchanged while aligning the
        advantage/loss computation with the RLSD training flow.
        """
        del num_items_in_batch

        # Batch-level prompt lengths define the sampled action span.
        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]

        # === STUDENT FORWARD ===
        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )
        student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]

        # Compute only sampled-token log-probabilities instead of materializing
        # the full vocab-sized log_softmax tensor, which is the main OOM source.
        student_logits = student_logits / self.temperature
        student_selected_logits = torch.gather(
            student_logits, dim=-1, index=sampled_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        student_log_denom = torch.logsumexp(student_logits, dim=-1)
        student_log_probs_sampled = student_selected_logits - student_log_denom
        del outputs_student, student_logits, student_selected_logits, student_log_denom
        empty_cache()

        # === TEACHER FORWARD ===
        if self.use_ema_teacher:
            adapter_context = self._ema_teacher_context(model)
        elif self.fixed_teacher and is_peft_model(model):
            adapter_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            adapter_context = nullcontext()

        with torch.no_grad(), adapter_context:
            outputs_teacher = model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
            )
            teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :]
            teacher_logits = teacher_logits / self.temperature
            teacher_selected_logits = torch.gather(
                teacher_logits, dim=-1, index=sampled_token_ids.unsqueeze(-1)
            ).squeeze(-1)
            teacher_log_denom = torch.logsumexp(teacher_logits, dim=-1)
            teacher_log_probs_sampled = teacher_selected_logits - teacher_log_denom
            del outputs_teacher, teacher_logits, teacher_selected_logits, teacher_log_denom
            empty_cache()

        # === RLSD-STYLE ADVANTAGE + LOSS ===
        raw_advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()
        advantage = raw_advantage
        response_mask = shifted_labels != -100
        low_advantage_flags = ((raw_advantage < self.low_advantage_threshold) & response_mask).any(dim=1)
        if "student_correct_flags" in inputs:
            student_correct_flags = inputs["student_correct_flags"].to(
                device=low_advantage_flags.device, dtype=torch.bool
            )
            low_advantage_flags = low_advantage_flags & student_correct_flags
        if low_advantage_flags.any():
            advantage = advantage.clone()
            advantage[low_advantage_flags] = 0.0

        response_advantage_min = torch.where(
            response_mask,
            raw_advantage.masked_fill(~response_mask, float("inf")),
            torch.full_like(raw_advantage, float("inf")),
        ).amin(dim=1)
        response_advantage_min = torch.where(
            torch.isinf(response_advantage_min),
            torch.zeros_like(response_advantage_min),
            response_advantage_min,
        )
        self._last_low_advantage_flags = low_advantage_flags.detach().cpu().tolist()
        self._last_response_advantages = response_advantage_min.detach().cpu().tolist()

        if shifted_labels is not None:
            mask = response_mask
            advantage = advantage[mask]
            student_log_probs_sampled = student_log_probs_sampled[mask]

        loss = -(advantage * student_log_probs_sampled).mean()

        del student_log_probs_sampled, teacher_log_probs_sampled, advantage
        empty_cache()

        if return_outputs:
            class MinimalOutput:
                def __init__(self):
                    self.loss = None

            minimal_output = MinimalOutput()
            minimal_output.loss = loss
            return loss, minimal_output
        return loss

    def generate_teacher_reasoning(
        self, model, teacher_reasoning_prompts, teacher_reasoning_attention_mask=None
    ):
        """Generate teacher's reasoning about the solution."""
        if self.use_vllm:
            # Use vLLM for fast reasoning generation
            return self._generate_teacher_reasoning_vllm(teacher_reasoning_prompts)
        else:
            generation_input_ids, generation_attention_mask = self._left_pad_for_generation(
                teacher_reasoning_prompts, teacher_reasoning_attention_mask
            )
            # Use transformers generation (slower)
            with torch.no_grad():
                # Temporarily enable KV cache
                original_use_cache = model.config.use_cache
                original_gen_use_cache = self.reasoning_generation_config.use_cache

                model.config.use_cache = True
                self.reasoning_generation_config.use_cache = True

                # If fixed_teacher=True, disable LoRA adapters
                adapter_context = (
                    self.accelerator.unwrap_model(model).disable_adapter()
                    if self.fixed_teacher and is_peft_model(model)
                    else nullcontext()
                )

                try:
                    with adapter_context:
                        reasoning_outputs = model.generate(
                            input_ids=generation_input_ids,
                            attention_mask=generation_attention_mask,
                            generation_config=self.reasoning_generation_config,
                            return_dict_in_generate=True,
                            use_cache=True,
                        )
                        reasoning_ids = reasoning_outputs.sequences
                finally:
                    model.config.use_cache = original_use_cache
                    self.reasoning_generation_config.use_cache = original_gen_use_cache

                return reasoning_ids

    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        """Generate on-policy outputs from student prompts only."""
        import time

        start_time = time.time()
        generation_input_ids, generation_attention_mask = self._left_pad_for_generation(
            inputs["student_prompts"],
            inputs.get("student_prompt_attention_mask", None),
        )

        # Temporarily enable KV cache for generation if it was disabled for training
        original_use_cache = model.config.use_cache
        original_gen_use_cache = generation_config.use_cache

        model.config.use_cache = True
        generation_config.use_cache = True

        print(f"\n{'='*80}")
        print(f"GENERATION DEBUG INFO:")
        print(f"  Model dtype: {model.dtype}")
        print(f"  Model config use_cache: {model.config.use_cache}")
        print(f"  Attention implementation: {getattr(model.config, '_attn_implementation', 'unknown')}")
        print(f"  Generation config use_cache: {generation_config.use_cache}")
        print(f"  Batch size: {generation_input_ids.shape[0]}")
        print(f"  Prompt length: {generation_input_ids.shape[1]}")
        print(f"  Max new tokens: {generation_config.max_new_tokens}")
        print(f"{'='*80}\n")

        # Generate output with respect to the student prompt only
        try:
            generated_outputs = model.generate(
                input_ids=generation_input_ids,
                attention_mask=generation_attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                use_cache=True,
            )
            # Get the generated token IDs
            generated_tokens = generated_outputs.sequences
        finally:
            # Restore original settings
            model.config.use_cache = original_use_cache
            generation_config.use_cache = original_gen_use_cache

        elapsed_time = time.time() - start_time
        num_prompts = generated_tokens.shape[0]
        total_completion_tokens = generated_tokens.shape[1] - generation_input_ids.shape[1]
        num_tokens = total_completion_tokens * num_prompts
        avg_completion_length = total_completion_tokens
        tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0
        print(
            f"generation done - elapsed time: {elapsed_time:.2f}s, prompts: {num_prompts}, total tokens: {num_tokens}, avg length: {avg_completion_length}, speed: {tokens_per_sec:.1f} tok/s"
        )

        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    @profiling_decorator
    def _generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id=None):
        """Generate on-policy outputs from student prompts using vLLM."""
        import time

        device = self.accelerator.device

        # Decode student prompts for vLLM (without special tokens - vLLM expects clean text)
        prompts_text_for_vllm = self.processing_class.batch_decode(
            inputs["student_prompts"],
            skip_special_tokens=True,
        )
        # Remove padding token text if it appears, as vLLM expects clean prompts
        if self.processing_class.pad_token:
            prompts_text_for_vllm = [
                p.replace(self.processing_class.pad_token, "") for p in prompts_text_for_vllm
            ]

        # Also decode prompts WITH special tokens for logging
        prompts_text_with_special = self.processing_class.batch_decode(
            inputs["student_prompts"],
            skip_special_tokens=False,
        )

        # system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        # target_system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        # prompts_text = [p.replace(target_system_prompt, system_prompt) for p in prompts_text]
        # Add system prompt to prompts

        max_completion_length = generation_config.max_new_tokens
        temperature = generation_config.temperature
        # vLLM uses top_k=-1 for no top_k, transformers uses 0 or None.
        top_k = generation_config.top_k if generation_config.top_k and generation_config.top_k > 0 else -1
        # top_p, repetition_penalty, min_p, presence_penalty are not directly in generation_config, get from trainer args
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0
        repetition_penalty = self.args.repetition_penalty if hasattr(self.args, "repetition_penalty") else 1.0
        min_p = self.args.min_p if hasattr(self.args, "min_p") else 0.0
        presence_penalty = self.args.presence_penalty if hasattr(self.args, "presence_penalty") else 0.0

        # Start timing for vLLM generation
        start_time = time.time()

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text_for_vllm)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,  # In GKD, we generate 1 completion per prompt from student
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_completion_length,
                    presence_penalty=presence_penalty,
                    guided_decoding_regex=self.vllm_guided_decoding_regex,
                )
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text_for_vllm),
                (self.accelerator.process_index + 1) * len(prompts_text_for_vllm),
            )
            completion_ids = completion_ids[process_slice]
        elif self.vllm_mode == "colocate":
            if self.vllm_guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(
                    backend="outlines", regex=self.vllm_guided_decoding_regex
                )
            else:
                guided_decoding = None
            sampling_params = SamplingParams(
                n=1,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_completion_length,
                presence_penalty=presence_penalty,
                guided_decoding=guided_decoding,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Gather prompts from all ranks in the TP group and flatten.
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                orig_size = len(prompts_text_for_vllm)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(
                    gathered_prompts, prompts_text_for_vllm, group=self.vllm_tp_group
                )
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text_for_vllm

            all_outputs = self.vllm_engine.generate(
                all_prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        # Calculate and print vLLM generation statistics
        elapsed_time = time.time() - start_time
        total_completion_tokens = sum(len(ids) for ids in completion_ids)
        num_prompts = len(completion_ids)
        avg_completion_length = total_completion_tokens / num_prompts if num_prompts > 0 else 0
        tokens_per_sec = total_completion_tokens / elapsed_time if elapsed_time > 0 else 0
        print(
            f"vLLM generation done - elapsed time: {elapsed_time:.2f}s, prompts: {num_prompts}, total tokens: {total_completion_tokens}, avg length: {avg_completion_length:.1f}, speed: {tokens_per_sec:.1f} tok/s"
        )

        # We need to combine prompt and completion for new_input_ids
        # Tokenize prompts again to get prompt_ids on the correct device and format
        # Use prompts_text_for_vllm (without special tokens) for tokenization since vLLM expects clean text
        # Ensure add_special_tokens=False as vLLM typically handles prompts as raw text
        # Calculate max_length for prompts, ensuring it's positive
        prompt_max_length = (
            max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
        )
        original_padding_side = self.processing_class.padding_side
        self.processing_class.padding_side = "left"
        try:
            prompt_tokenized = self.processing_class(
                prompts_text_for_vllm,
                return_tensors="pt",
                padding="longest",
                truncation=True if prompt_max_length else False,
                max_length=prompt_max_length,
                add_special_tokens=False,
            ).to(device)
        finally:
            self.processing_class.padding_side = original_padding_side
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        # Manually pad/truncate completions to max_completion_length length before using pad function
        padded_completion_ids_list = []
        for completion_tensor in completion_ids_tensors:
            if len(completion_tensor) > max_completion_length:
                # Truncate if longer than max_completion_length
                padded_completion_ids_list.append(completion_tensor[:max_completion_length])
            elif len(completion_tensor) < max_completion_length:
                # Pad if shorter than max_completion_length
                padding_needed = max_completion_length - len(completion_tensor)
                padded_tensor = torch.cat(
                    [
                        completion_tensor,
                        torch.full(
                            (padding_needed,), pad_token_id, device=device, dtype=completion_tensor.dtype
                        ),
                    ]
                )
                padded_completion_ids_list.append(padded_tensor)
            else:
                # Already the right length
                padded_completion_ids_list.append(completion_tensor)

        # Now all tensors are the same length, so we can stack them
        padded_completion_ids = torch.stack(padded_completion_ids_list)

        # Ensure prompt_ids and padded_completion_ids are 2D
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if padded_completion_ids.ndim == 1:
            padded_completion_ids = padded_completion_ids.unsqueeze(0)

        new_input_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)

        new_attention_mask = torch.ones_like(new_input_ids, device=device)
        new_labels = new_input_ids.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[new_input_ids == pad_token_id] = 0

        # Extract completion texts from the generated completion IDs
        completion_texts = []
        for comp_ids in completion_ids:
            completion_text = self.processing_class.decode(comp_ids, skip_special_tokens=False)
            completion_texts.append(completion_text)

        return new_input_ids, new_attention_mask, new_labels, prompts_text_with_special, completion_texts

    def _generate_teacher_reasoning_vllm(
        self, teacher_reasoning_prompts, teacher_reasoning_attention_mask=None
    ):
        """Generate teacher's reasoning using vLLM."""
        import time

        device = self.accelerator.device

        # Decode prompts for vLLM
        prompts_text = self.processing_class.batch_decode(
            teacher_reasoning_prompts,
            skip_special_tokens=True,
        )
        if self.processing_class.pad_token:
            prompts_text = [p.replace(self.processing_class.pad_token, "") for p in prompts_text]

        max_reasoning_length = self.reasoning_generation_config.max_new_tokens
        temperature = self.reasoning_generation_config.temperature
        top_k = (
            self.reasoning_generation_config.top_k
            if self.reasoning_generation_config.top_k and self.reasoning_generation_config.top_k > 0
            else -1
        )
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0

        start_time = time.time()

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_reasoning_length,
                )
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text),
                (self.accelerator.process_index + 1) * len(prompts_text),
            )
            completion_ids = completion_ids[process_slice]

        elif self.vllm_mode == "colocate":
            sampling_params = SamplingParams(
                n=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_reasoning_length,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                orig_size = len(prompts_text)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.vllm_tp_group)
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text

            all_outputs = self.vllm_engine.generate(
                all_prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)

        elapsed_time = time.time() - start_time
        total_tokens = sum(len(ids) for ids in completion_ids)
        num_prompts = len(completion_ids)
        print(
            f"vLLM teacher reasoning generation done - elapsed: {elapsed_time:.2f}s, prompts: {num_prompts}, tokens: {total_tokens}, speed: {total_tokens/elapsed_time:.1f} tok/s"
        )

        # Combine prompt + completion
        original_padding_side = self.processing_class.padding_side
        self.processing_class.padding_side = "left"
        try:
            prompt_tokenized = self.processing_class(
                prompts_text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                add_special_tokens=False,
            ).to(device)
        finally:
            self.processing_class.padding_side = original_padding_side
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        padded_completions = pad(
            completion_ids_tensors, padding_value=self.processing_class.pad_token_id, padding_side="right"
        )

        reasoning_ids = torch.cat([prompt_ids, padded_completions], dim=1)

        return reasoning_ids

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with student vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            # recurse into the child
            self._sync_fsdp_params_to_vllm(child_module, prefix=child_prefix, visited=visited)

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = (
                            self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                        )
                        llm_model.load_weights([(full_name, param.data)])

    def _move_model_to_vllm(self):
        """Synchronize student model weights to vLLM engine."""
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["weights"])

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = (
                                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            )
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                # use memory-efficient post-order traversal for FSDP
                self._sync_fsdp_params_to_vllm(self.model)
            else:
                # For DeepSpeed ZeRO-3, gather each parameter individually like GRPO trainer
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = (
                                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            )
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.vllm_engine.reset_prefix_cache()

    def _wake_vllm_if_needed(self):
        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["kv_cache"])

    def _save_generation_outputs(self, step: int):
        """Save generation outputs to disk."""
        if not self.accelerator.is_main_process:
            return

        if len(self._generation_outputs_buffer) == 0:
            return

        import json
        from pathlib import Path

        # Create generations directory in output_dir
        generations_dir = Path(self.args.output_dir) / "generations"
        generations_dir.mkdir(parents=True, exist_ok=True)

        # Save to JSON file
        output_file = generations_dir / f"generations_step_{step}.json"

        output_data = {
            "step": step,
            "num_samples": len(self._generation_outputs_buffer),
            "generations": self._generation_outputs_buffer,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"Saved {len(self._generation_outputs_buffer)} generation outputs to:")
        print(f"  {output_file}")
        print(f"{'='*80}\n")

        # Clear buffer after saving
        self._generation_outputs_buffer.clear()

    @staticmethod
    def _flatten_gathered_records(records):
        if not records:
            return []
        if isinstance(records, list) and records and isinstance(records[0], dict):
            return records
        flat = []
        for item in records:
            if isinstance(item, list):
                flat.extend(item)
            elif isinstance(item, dict):
                flat.append(item)
        return flat

    def _append_jsonl_records(self, path: Path, records: list[dict[str, Any]]) -> None:
        if not records or not self.accelerator.is_main_process:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _append_sample_text_log(self, title: str, payload: dict[str, Any]) -> None:
        if not self.accelerator.is_main_process:
            return
        self.sample_text_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.sample_text_log_path, "a", encoding="utf-8") as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"{title}\n")
            f.write(f"{'=' * 80}\n")
            for key, value in payload.items():
                f.write(f"{key}:\n{value}\n\n")

    def _load_jsonl_records(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _write_jsonl_records(self, path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _window_start_step(step: int, window_size: int) -> int:
        return max(1, step - window_size + 1)

    def _write_json_payload(self, path: Path, payload: dict[str, Any]) -> None:
        if not self.accelerator.is_main_process:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _group_prompt_indices(prompt_texts: list[str]) -> list[list[int]]:
        if not prompt_texts:
            return []
        groups: list[list[int]] = []
        current_group = [0]
        for idx in range(1, len(prompt_texts)):
            if prompt_texts[idx] == prompt_texts[current_group[0]]:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
        groups.append(current_group)
        return groups

    def _record_question_response_window(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prompt_texts: list[str],
        completion_texts: list[str],
    ) -> None:
        prompt_groups = self._group_prompt_indices(prompt_texts)
        if not prompt_groups:
            return

        teacher_prompt_rows = [group[0] for group in prompt_groups]
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            teacher_responses = self._generate_teacher_completions(
                unwrapped_model,
                inputs["teacher_prompts"][teacher_prompt_rows],
                inputs["teacher_prompt_attention_mask"][teacher_prompt_rows],
                int(inputs["teacher_prompt_length"]),
            )

        solution_rows = [inputs["solutions"][group[0]] for group in prompt_groups]
        student_rewards = reward_correctness_from_solutions(completion_texts, inputs["solutions"])
        teacher_rewards = reward_correctness_from_solutions(teacher_responses, solution_rows)

        records = []
        for question_idx, group in enumerate(prompt_groups):
            sample_idx = group[0]
            records.append(
                {
                    "step": int(self.state.global_step),
                    "question_index_in_step": int(question_idx),
                    "problem": inputs["problems"][sample_idx],
                    "prompt": prompt_texts[sample_idx],
                    "student_responses": [completion_texts[idx] for idx in group],
                    "student_rewards": [float(student_rewards[idx]) for idx in group],
                    "teacher_response": teacher_responses[question_idx],
                    "teacher_reward": float(teacher_rewards[question_idx]),
                }
            )

        gathered_records = gather_object(records)
        gathered_records = self._flatten_gathered_records(gathered_records)
        if self.accelerator.is_main_process and gathered_records:
            self._question_response_buffer.extend(gathered_records)

    def _flush_question_response_window(self, step: int) -> None:
        if not self.accelerator.is_main_process or not self._question_response_buffer:
            return
        start_step = self._window_start_step(step, self._window_save_frequency)
        out_path = self.question_response_dir / f"question_rollouts_step_{start_step:06d}_{step:06d}.json"
        payload = {
            "step_start": start_step,
            "step_end": int(step),
            "num_questions": len(self._question_response_buffer),
            "records": self._question_response_buffer,
        }
        self._write_json_payload(out_path, payload)
        self._question_response_buffer.clear()

    def _flush_sft_window_buffer(self, step: int) -> None:
        if not self.accelerator.is_main_process or not self._sft_window_buffer:
            return
        start_step = self._window_start_step(step, self._window_save_frequency)
        out_path = self.sft_windows_dir / f"sft_examples_step_{start_step:06d}_{step:06d}.jsonl"
        self._write_jsonl_records(out_path, self._sft_window_buffer)
        self._append_jsonl_records(self.pending_sft_path, self._sft_window_buffer)
        self._sft_window_buffer.clear()

    def _flush_window_buffers(self, step: int) -> None:
        self._flush_question_response_window(step)
        self._flush_sft_window_buffer(step)

    def _build_problem_prompt(self, problem: str) -> str:
        user_message = (
            f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        )
        return self.processing_class.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _build_sft_batch(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        original_padding_side = self.processing_class.padding_side
        self.processing_class.padding_side = "right"
        prompts = [self._build_problem_prompt(example["problem"]) for example in examples]
        responses = [example["response"] for example in examples]
        full_texts = [prompt + response for prompt, response in zip(prompts, responses)]

        try:
            prompt_ids = self.processing_class(
                prompts,
                padding=False,
                truncation=True,
                max_length=self.args.max_length,
            )["input_ids"]
            prompt_lengths = [len(ids) for ids in prompt_ids]

            encoded = self.processing_class(
                full_texts,
                padding="longest",
                truncation=True,
                max_length=self.args.max_length,
                return_tensors="pt",
            )
        finally:
            self.processing_class.padding_side = original_padding_side

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = input_ids.clone()
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100
        labels[attention_mask == 0] = -100
        if self.processing_class.pad_token_id is not None:
            labels[input_ids == self.processing_class.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _sft_collate_fn(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return self._build_sft_batch(examples)

    def _left_pad_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            if self.processing_class.pad_token_id is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            else:
                attention_mask = (input_ids != self.processing_class.pad_token_id).long()

        token_seqs = []
        mask_seqs = []
        for ids, mask in zip(input_ids, attention_mask):
            valid_ids = ids[mask.bool()]
            token_seqs.append(valid_ids)
            mask_seqs.append(torch.ones_like(valid_ids, dtype=torch.long))

        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processing_class.eos_token_id

        padded_ids = pad(token_seqs, padding_value=pad_token_id, padding_side="left")
        padded_mask = pad(mask_seqs, padding_value=0, padding_side="left")
        return padded_ids, padded_mask

    def _repeat_generation_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self.num_generations <= 1:
            return inputs

        repeated_inputs: dict[str, Any] = {}
        repeatable_tensor_keys = {
            "student_prompts",
            "student_prompt_attention_mask",
            "teacher_prompts",
            "teacher_prompt_attention_mask",
            "teacher_reasoning_prompts",
            "teacher_reasoning_attention_mask",
            "teacher_transition_tokens",
        }
        repeatable_list_keys = {"problems", "solutions"}

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and key in repeatable_tensor_keys:
                repeated_inputs[key] = value.repeat_interleave(self.num_generations, dim=0)
            elif isinstance(value, list) and key in repeatable_list_keys:
                repeated_inputs[key] = [item for item in value for _ in range(self.num_generations)]
            else:
                repeated_inputs[key] = value

        return repeated_inputs

    def _generate_teacher_completions(
        self,
        model: nn.Module,
        teacher_prompts: torch.Tensor,
        teacher_attention_mask: torch.Tensor,
        teacher_prompt_length: int,
    ) -> list[str]:
        if teacher_prompts.shape[0] == 0:
            return []

        generation_input_ids, generation_attention_mask = self._left_pad_for_generation(
            teacher_prompts, teacher_attention_mask
        )

        with torch.no_grad():
            original_use_cache = model.config.use_cache
            original_gen_use_cache = self.generation_config.use_cache
            model.config.use_cache = True
            self.generation_config.use_cache = True

            if self.use_ema_teacher:
                adapter_context = self._ema_teacher_context(model)
            elif self.fixed_teacher and is_peft_model(model):
                adapter_context = self.accelerator.unwrap_model(model).disable_adapter()
            else:
                adapter_context = nullcontext()

            try:
                with adapter_context:
                    generated = model.generate(
                        input_ids=generation_input_ids,
                        attention_mask=generation_attention_mask,
                        generation_config=self.generation_config,
                        return_dict_in_generate=True,
                        use_cache=True,
                    )
                    generated_ids = generated.sequences
            finally:
                model.config.use_cache = original_use_cache
                self.generation_config.use_cache = original_gen_use_cache

        completion_ids = generated_ids[:, generation_input_ids.shape[1] :]
        return self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)

    def _record_low_advantage_examples(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prompt_texts: list[str],
        completion_texts: list[str],
    ) -> None:
        low_mask = self._last_low_advantage_flags
        response_advantages = self._last_response_advantages
        if low_mask is None or response_advantages is None:
            return

        flagged_indices = [idx for idx, is_flagged in enumerate(low_mask) if is_flagged]
        if not flagged_indices:
            return

        correctness_rewards = reward_correctness_from_solutions(completion_texts, inputs["solutions"])
        eligible_indices = [idx for idx in flagged_indices if correctness_rewards[idx] > 0.5]

        student_records = []
        for idx in flagged_indices:
            student_records.append(
                {
                    "step": int(self.state.global_step),
                    "problem": inputs["problems"][idx],
                    "student_prompt": prompt_texts[idx],
                    "student_response": completion_texts[idx],
                    "student_reward": float(correctness_rewards[idx]),
                    "response_advantage_min": float(response_advantages[idx]),
                    "threshold": float(self.low_advantage_threshold),
                }
            )

        gathered_student = gather_object(student_records)
        gathered_student = self._flatten_gathered_records(gathered_student)
        self._append_jsonl_records(self.low_adv_student_path, gathered_student)

        if not eligible_indices:
            return

        sft_records = []
        for sample_idx in eligible_indices:
            sft_records.append(
                {
                    "step": int(self.state.global_step),
                    "problem": inputs["problems"][sample_idx],
                    "response": completion_texts[sample_idx],
                    "source": "student_correct_response",
                    "student_reward": float(correctness_rewards[sample_idx]),
                    "response_advantage_min": float(response_advantages[sample_idx]),
                    "threshold": float(self.low_advantage_threshold),
                }
            )

        teacher_group_indices = []
        last_problem = None
        for sample_idx in eligible_indices:
            problem = inputs["problems"][sample_idx]
            if problem != last_problem:
                teacher_group_indices.append(sample_idx)
                last_problem = problem

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            teacher_completions = self._generate_teacher_completions(
                unwrapped_model,
                inputs["teacher_prompts"][teacher_group_indices],
                inputs["teacher_prompt_attention_mask"][teacher_group_indices],
                int(inputs["teacher_prompt_length"]),
            )

        teacher_rewards = reward_correctness_from_solutions(
            teacher_completions,
            [inputs["solutions"][sample_idx] for sample_idx in teacher_group_indices],
        )

        for local_idx, sample_idx in enumerate(teacher_group_indices):
            sft_records.append(
                {
                    "step": int(self.state.global_step),
                    "problem": inputs["problems"][sample_idx],
                    "response": teacher_completions[local_idx],
                    "source": "teacher_response",
                    "teacher_reward": float(teacher_rewards[local_idx]),
                    "threshold": float(self.low_advantage_threshold),
                }
            )

        gathered_sft = gather_object(sft_records)
        gathered_sft = self._flatten_gathered_records(gathered_sft)
        if self.accelerator.is_main_process and gathered_sft:
            self._sft_window_buffer.extend(gathered_sft)

    def _run_periodic_sft(self) -> None:
        if self._sft_running:
            return

        self.accelerator.wait_for_everyone()
        pending_examples = self._load_jsonl_records(self.pending_sft_path) if self.accelerator.is_main_process else None
        pending_examples = broadcast_object_list([pending_examples], from_process=0)[0]
        if not pending_examples:
            if self.accelerator.is_main_process:
                print("Periodic SFT skipped: no pending remediation examples.")
            return

        if len(pending_examples) < self.periodic_sft_dataset_size:
            if self.accelerator.is_main_process:
                print(
                    "Periodic SFT skipped: "
                    f"{len(pending_examples)}/{self.periodic_sft_dataset_size} samples collected."
                )
            return

        current_sft_examples = pending_examples[: self.periodic_sft_dataset_size]
        remaining_examples = pending_examples[self.periodic_sft_dataset_size :]

        if self.periodic_sft_max_samples > 0:
            current_sft_examples = current_sft_examples[: self.periodic_sft_max_samples]

        self._sft_running = True
        original_lr = [group["lr"] for group in self.optimizer.param_groups]
        for group in self.optimizer.param_groups:
            group["lr"] = self.periodic_sft_learning_rate

        try:
            print(f"\n{'='*80}")
            print(
                f"STARTING PERIODIC SFT AT STEP {self.state.global_step} "
                f"WITH {len(current_sft_examples)} REMEDIATION EXAMPLES"
            )
            print(f"{'='*80}\n")

            sft_loader = DataLoader(
                current_sft_examples,
                batch_size=self.periodic_sft_batch_size,
                shuffle=True,
                collate_fn=self._sft_collate_fn,
            )
            self.model.train()

            for epoch_idx in range(self.periodic_sft_epochs):
                epoch_loss = 0.0
                step_count = 0
                for batch in sft_loader:
                    batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                    self.optimizer.zero_grad(set_to_none=True)
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    epoch_loss += float(loss.detach())
                    step_count += 1

                avg_loss = epoch_loss / max(1, step_count)
                print(
                    f"Periodic SFT epoch {epoch_idx + 1}/{self.periodic_sft_epochs} "
                    f"completed, avg_loss={avg_loss:.4f}"
                )

            if self.accelerator.is_main_process:
                sft_checkpoint = self.sft_checkpoint_dir / f"step_{self.state.global_step}"
                self.save_model(str(sft_checkpoint))
                metadata = {
                    "step": int(self.state.global_step),
                    "num_examples": len(current_sft_examples),
                    "epochs": int(self.periodic_sft_epochs),
                    "learning_rate": float(self.periodic_sft_learning_rate),
                    "target_dataset_size": int(self.periodic_sft_dataset_size),
                }
                with open(sft_checkpoint / "remediation_sft_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                archive_path = self.remediation_dir / f"consumed_sft_examples_step_{self.state.global_step}.jsonl"
                self._write_jsonl_records(archive_path, current_sft_examples)
                self._write_jsonl_records(self.pending_sft_path, remaining_examples)

            if self.use_vllm:
                self._move_model_to_vllm()

            if self.accelerator.is_main_process:
                print(f"\n{'='*80}")
                print(f"PERIODIC SFT COMPLETED AT STEP {self.state.global_step}")
                print(f"SFT checkpoint saved to: {sft_checkpoint}")
                print(f"{'='*80}\n")
        finally:
            self.accelerator.wait_for_everyone()
            for group, lr in zip(self.optimizer.param_groups, original_lr):
                group["lr"] = lr
            self.optimizer.zero_grad(set_to_none=True)
            self._sft_running = False
            empty_cache()

    def _maybe_run_periodic_sft(self) -> None:
        if self.periodic_sft_dataset_size <= 0:
            return
        if self.state.global_step <= 0:
            return
        self._run_periodic_sft()

    @profiling_decorator
    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        """
        Perform a training step with self-distillation.

        If reason_first=True:
        1. Generate teacher's reasoning about the solution
        2. Append reasoning to teacher prompt
        3. Generate completions from student prompts
        4. Compute JSD loss

        Otherwise:
        1. Generate completions from student prompts
        2. Construct full sequences for both student and teacher with the generation
        3. Compute JSD loss on the generation tokens
        """
        on_policy = True

        # === REASONING PHASE (if enabled) ===
        if self.reason_first:
            print(f"\n{'='*80}")
            print("REASONING PHASE: Teacher analyzing solution...")
            print(f"{'='*80}\n")

            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                # Generate teacher's reasoning
                teacher_reasoning_ids = self.generate_teacher_reasoning(
                    unwrapped_model,
                    inputs["teacher_reasoning_prompts"],
                    inputs.get("teacher_reasoning_attention_mask"),
                )

                # Decode reasoning
                reasoning_prompt_len = inputs["teacher_reasoning_prompt_length"]
                reasoning_completions = teacher_reasoning_ids[:, reasoning_prompt_len:]
                reasoning_texts = self.processing_class.batch_decode(
                    reasoning_completions, skip_special_tokens=True
                )

                # Occasionally print reasoning
                if random.random() < 0.01:
                    sample_idx = random.randint(0, len(reasoning_texts) - 1)
                    sample_prompt = self.processing_class.decode(
                        inputs["teacher_reasoning_prompts"][sample_idx], skip_special_tokens=False
                    )
                    self._append_sample_text_log(
                        f"TEACHER REASONING SAMPLE (Step {self.state.global_step})",
                        {
                            "PROMPT": sample_prompt,
                            "REASONING": reasoning_texts[sample_idx],
                        },
                    )

                # Update teacher prompts with reasoning
                # Construct: [teacher_reasoning_prompt][reasoning][transition_to_teaching]
                teacher_prompts_with_reasoning = torch.cat(
                    [
                        inputs["teacher_reasoning_prompts"],
                        reasoning_completions,
                        inputs["teacher_transition_tokens"],
                    ],
                    dim=1,
                )

                # Update inputs with new teacher prompts
                inputs["teacher_prompts"] = teacher_prompts_with_reasoning
                teacher_attention_mask = torch.ones_like(teacher_prompts_with_reasoning)
                if self.processing_class.pad_token_id is not None:
                    teacher_attention_mask[
                        teacher_prompts_with_reasoning == self.processing_class.pad_token_id
                    ] = 0
                inputs["teacher_prompt_attention_mask"] = teacher_attention_mask
                inputs["teacher_prompt_length"] = teacher_prompts_with_reasoning.shape[1]

        inputs = self._repeat_generation_inputs(inputs)

        # === GENERATION PHASE ===
        if self.use_vllm:
            self._wake_vllm_if_needed()
            result = self._generate_on_policy_outputs_vllm(
                inputs, self.generation_config, self.processing_class.pad_token_id
            )
            generated_ids, generated_attention_mask, _, prompt_texts, completion_texts = result
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                result = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
                generated_ids, generated_attention_mask, _ = result
                # Decode for logging
                prompt_texts = self.processing_class.batch_decode(
                    inputs["student_prompts"], skip_special_tokens=False
                )
                student_prompt_len = inputs["student_prompt_length"]
                completion_ids = generated_ids[:, student_prompt_len:]
                completion_texts = self.processing_class.batch_decode(
                    completion_ids, skip_special_tokens=False
                )

        # Get batch-level student prompt length
        student_prompt_len = inputs["student_prompt_length"]

        # Extract generation part (same slice for all examples since prompts are padded)
        generation_ids = generated_ids[:, student_prompt_len:]

        # Construct student full sequence: [student_prompt][generation]
        inputs["student_input_ids"] = generated_ids
        inputs["student_attention_mask"] = generated_attention_mask

        # Construct teacher full sequence: [teacher_prompt][generation]
        teacher_prompts = inputs["teacher_prompts"]
        teacher_full_ids = torch.cat([teacher_prompts, generation_ids], dim=1)

        # Create attention mask for teacher
        teacher_attention_mask = torch.ones_like(teacher_full_ids)
        if self.processing_class.pad_token_id is not None:
            teacher_attention_mask[teacher_full_ids == self.processing_class.pad_token_id] = 0

        inputs["teacher_input_ids"] = teacher_full_ids
        inputs["teacher_attention_mask"] = teacher_attention_mask

        # Create labels for generation tokens
        # Generation inputs are left-padded before sampling, so the whole prompt span is
        # the shared prefix of length `student_prompt_len`.
        labels = generated_ids.clone()
        labels[:, :student_prompt_len] = -100

        if self.processing_class.pad_token_id is not None:
            labels[labels == self.processing_class.pad_token_id] = -100

        inputs["labels"] = labels

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompt_texts))
        self._textual_logs["completion"].extend(gather_object(completion_texts))
        correctness_rewards = reward_correctness_from_solutions(completion_texts, inputs["solutions"])
        inputs["student_correct_flags"] = torch.tensor(
            [reward > 0.5 for reward in correctness_rewards],
            device=generated_ids.device,
            dtype=torch.bool,
        )

        # Collect generation outputs for saving
        for prompt, completion in zip(prompt_texts, completion_texts):
            self._generation_outputs_buffer.append(
                {"step": self.state.global_step, "prompt": prompt, "completion": completion}
            )

        # Occasionally print student's generation with 10% probability
        if random.random() < 0.9:
            sample_idx = random.randint(0, len(prompt_texts) - 1)
            self._append_sample_text_log(
                f"STUDENT GENERATION SAMPLE (Step {self.state.global_step})",
                {
                    "PROMPT": prompt_texts[sample_idx],
                    "COMPLETION": completion_texts[sample_idx],
                },
            )

        self._record_question_response_window(model, inputs, prompt_texts, completion_texts)
        loss = super().training_step(model, inputs, num_items_in_batch)

        self._record_low_advantage_examples(model, inputs, prompt_texts, completion_texts)

        # Save generation outputs every N steps
        if (
            self.state.global_step > 0
            and self.state.global_step % self._generation_save_frequency == 0
            and self.accelerator.sync_gradients
        ):
            self._save_generation_outputs(self.state.global_step)
            self._flush_window_buffers(self.state.global_step)

        loss_scalar = float(loss.detach())
        ga = max(1, int(self.args.gradient_accumulation_steps))
        step_equiv = 1.0 / ga

        if on_policy:
            self._on_policy_loss_total += loss_scalar
            self._on_policy_step_equiv += step_equiv
        else:
            self._off_policy_loss_total += loss_scalar
            self._off_policy_step_equiv += step_equiv
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        if mode == "train":
            device = self.accelerator.device if hasattr(self.accelerator, "device") else torch.device("cpu")
            # Track on/off-policy loss statistics
            vec = torch.tensor(
                [
                    self._on_policy_loss_total,
                    self._off_policy_loss_total,
                    self._on_policy_step_equiv,
                    self._off_policy_step_equiv,
                ],
                dtype=torch.float64,
                device=device,
            )

            # Sum across processes so we mirror Trainer's distributed reduction
            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            (
                on_sum,
                off_sum,
                on_eq,
                off_eq,
            ) = vec.tolist()

            # Compute category averages over the *same window* as Trainer's logs
            # (avoid div-by-zero if, e.g., no on-policy steps in the window)
            if on_eq > 0:
                logs["on_policy_loss"] = round(on_sum / on_eq, 4)
            if off_eq > 0:
                logs["off_policy_loss"] = round(off_sum / off_eq, 4)

            # Reset window accumulators after logging (just like Trainer resets its window)
            self._on_policy_loss_total = self._off_policy_loss_total = 0.0
            self._on_policy_step_equiv = self._off_policy_step_equiv = 0.0

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if (
            self.accelerator.is_main_process
            and self.log_completions
            and ((self.state.global_step % self.log_completion_steps) == 0)
        ):

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                if self.num_completions_to_print and len(df) > 0:
                    df = df.sample(n=self.num_completions_to_print, random_state=42)
                wandb.log({"completions": wandb.Table(dataframe=df)})
