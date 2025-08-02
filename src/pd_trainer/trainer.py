import logging
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from transformers.trainer import _is_peft_model
from trl import SFTTrainer
from trl.trainer.utils import pad


def concat_forward(
    model, prompt_ids, response_ids, prompt_mask, response_mask, return_outputs=False
):
    input_ids = torch.cat([prompt_ids, response_ids], dim=-1)
    input_mask = torch.cat([prompt_mask, response_mask], dim=-1)
    outputs = model(input_ids=input_ids, attention_mask=input_mask)
    logits = outputs.logits[:, prompt_ids.size(-1) - 1 : -1, :]
    return (logits, outputs) if return_outputs else logits


class PDDataCollator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def tokenize_row(self, row):
        tools = row.get("tools")

        is_conversational_format = isinstance(row["prompt"], list)
        if is_conversational_format:
            prompt = self.tokenizer.apply_chat_template(
                row["prompt"],
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            )
            teacher_prompt = self.tokenizer.apply_chat_template(
                row["teacher_prompt"],
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            )
            teacher_prompt_completion = self.tokenizer.apply_chat_template(
                row["teacher_prompt"] + row["completion"],
                tools=tools,
                tokenize=False,
            )
        else:
            prompt = row["prompt"]
            teacher_prompt = row["teacher_prompt"]
            teacher_prompt_completion = row["teacher_prompt"] + row["completion"]

        prompt_ids = (
            self.tokenizer(prompt, return_tensors="pt")
            .input_ids.squeeze(0)
            .to(self.model.device)
        )
        teacher_prompt_ids = (
            self.tokenizer(teacher_prompt, return_tensors="pt")
            .input_ids.squeeze(0)
            .to(self.model.device)
        )
        teacher_prompt_completion_ids = (
            self.tokenizer(teacher_prompt_completion, return_tensors="pt")
            .input_ids.squeeze(0)
            .to(self.model.device)
        )
        completion_ids = teacher_prompt_completion_ids[
            ..., teacher_prompt_ids.size(-1) :
        ]

        tokenized = {
            "prompt_ids": prompt_ids,
            "teacher_prompt_ids": teacher_prompt_ids,
            "completion_ids": completion_ids,
        }

        return tokenized

    def __call__(self, dataset):
        collated = [self.tokenize_row(row) for row in dataset]

        assert len(collated) > 0, "No data."

        def pad_ids(key, padding_side="right"):
            return pad(
                [record[key] for record in collated],
                padding_value=self.tokenizer.pad_token_id,
                padding_side=padding_side,
            )

        def pad_mask(key, padding_value=0, padding_side="right"):
            return pad(
                [torch.ones_like(record[key]) for record in collated],
                padding_value=padding_value,
                padding_side=padding_side,
            )

        prompt_ids = pad_ids("prompt_ids", padding_side="left")
        prompt_mask = pad_mask("prompt_ids", padding_side="left")
        teacher_prompt_ids = pad_ids("teacher_prompt_ids", padding_side="left")
        teacher_prompt_mask = pad_mask("teacher_prompt_ids", padding_side="left")
        completion_ids = pad_ids("completion_ids")
        completion_mask = pad_mask("completion_ids")

        with torch.inference_mode():
            if _is_peft_model(self.model):
                model_context = self.model.disable_adapter()
            else:
                model_context = nullcontext()

            with model_context:
                teacher_completion_logits = concat_forward(
                    self.model,
                    teacher_prompt_ids,
                    completion_ids,
                    teacher_prompt_mask,
                    completion_mask,
                ).detach()

        if collated[0].get("labels"):
            labels = pad_mask("labels", padding_value=-100, padding_side="right").to(
                "cpu"
            )
        else:
            labels = None

        inputs = {
            "prompt_ids": prompt_ids.to("cpu"),
            "prompt_mask": prompt_mask.to("cpu"),
            "teacher_prompt_ids": teacher_prompt_ids.to("cpu"),
            "teacher_prompt_mask": teacher_prompt_mask.to("cpu"),
            "completion_ids": completion_ids.to("cpu"),
            "completion_mask": completion_mask.to("cpu"),
            "teacher_completion_logits": teacher_completion_logits.to("cpu"),
            "labels": labels,
        }

        return inputs


class PDTrainer(SFTTrainer):

    _tag_names = ["trl", "pd"]

    def __init__(self, *args, **kwargs):
        is_model_str = False

        if kwargs.get("data_collator") is None:
            ref_model = kwargs.pop("ref_model", None)
            if ref_model is not None:
                model = ref_model
            else:
                model = kwargs.get("model") or args[0]
                is_model_str = isinstance(model, str)
                is_peft_model = kwargs.get("peft_config") is not None or _is_peft_model(
                    model
                )

                # LoRA allows using the base model as the reference model.
                if not is_peft_model:
                    logging.warning(
                        "Using training model itself as the reference model is dangerous."
                    )

            tokenizer = kwargs.get("processing_class")
            kwargs["data_collator"] = PDDataCollator(model, tokenizer)

        super().__init__(*args, **kwargs)

        if is_model_str:
            kwargs["data_collator"].model = self.model

    def _prepare_dataset(self, dataset, *args, **kwargs):
        return dataset

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt",
                "teacher_prompt",
                "completion",
                "tools",
            ]

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        student_completion_logits, student_outputs = concat_forward(
            model=model,
            prompt_ids=inputs["prompt_ids"],
            response_ids=inputs["completion_ids"],
            prompt_mask=inputs["prompt_mask"],
            response_mask=inputs["completion_mask"],
            return_outputs=True,
        )

        teacher_completion_logits = inputs["teacher_completion_logits"]

        p_teacher = teacher_completion_logits.softmax(dim=-1)
        p_student = student_completion_logits.softmax(dim=-1)

        # TODO: Read `distill_topk` from config
        distill_topk = 4
        _, idx_topk = torch.topk(teacher_completion_logits, distill_topk, dim=-1)

        tgt_teacher = p_teacher.gather(-1, idx_topk)
        rest_teacher = torch.clamp_min(1 - tgt_teacher.sum(dim=-1, keepdim=True), 1e-10)
        tgt_teacher = torch.cat((tgt_teacher, rest_teacher), dim=-1)
        tgt_student = p_student.gather(-1, idx_topk)
        rest_student = torch.clamp_min(1 - tgt_student.sum(dim=-1, keepdim=True), 1e-10)
        tgt_student = torch.cat((tgt_student, rest_student), dim=-1)

        kl_loss = F.kl_div(
            input=tgt_student.log(), target=tgt_teacher, reduction="none"
        ).sum(dim=-1)

        if inputs.get("labels") is None:
            completion_loss_mask = inputs["completion_mask"]
        else:
            completion_loss_mask = inputs["labels"] == -100

        loss = ((kl_loss * completion_loss_mask.float()).sum()) / (
            completion_loss_mask.sum()
        )

        with torch.no_grad():
            logp_teacher = F.log_softmax(teacher_completion_logits, dim=-1)
            logp_student = F.log_softmax(student_completion_logits, dim=-1)

            tgt_ids = inputs["completion_ids"]

            nll_teacher = -logp_teacher.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
            nll_student = -logp_student.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)

            mask = completion_loss_mask
            nll_teacher = nll_teacher * mask
            nll_student = nll_student * mask
            token_cnt = mask.sum(dim=-1)

            teacher_ppl_per_sample = torch.exp(nll_teacher.sum(dim=-1) / token_cnt)
            student_ppl_per_sample = torch.exp(nll_student.sum(dim=-1) / token_cnt)

            self.log(
                {
                    "teacher_ppl": teacher_ppl_per_sample.mean().item(),
                    "student_ppl": student_ppl_per_sample.mean().item(),
                }
            )

        return (loss, student_outputs) if return_outputs else loss
