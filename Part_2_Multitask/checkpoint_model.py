import transformers
import torch
import os


def save_model(model_name, multitask_model):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    for task_name in ["rte", "stsb", "commonsense_qa"]:
        os.makedirs(f"./{task_name}_model", exist_ok=True)
        multitask_model.taskmodels_dict[task_name].config.to_json_file(
            f"./{task_name}_model/config.json"
        )
        torch.save(
            multitask_model.taskmodels_dict[task_name].state_dict(),
            f"./{task_name}_model/pytorch_model.bin",
        )
        tokenizer.save_pretrained(f"./{task_name}_model/")
