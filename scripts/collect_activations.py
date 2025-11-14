import os
import sys
import os
import sys

# Add the src directory to sys.path for imports to work
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
SRC_DIR = os.path.normpath(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse

from src.utils import setup_model, get_device_info
import torch
import json


def main(model_id, prompts_path, objects_path, colorlist_path, output_dir):
    model, tokenizer = setup_model(model_id)

    prompt_templates = json.load(open(prompts_path))
    objects = json.load(open(objects_path))
    colorlist = json.load(open(colorlist_path))

    # create a list of prompts, using the prompt templates and the colors
    prompts = [prompt_template.format(object=object) for prompt_template in prompt_templates for object in objects]

    print(prompts)

    # sum_acts[color][layer] = torch.zeros(len(colors), model.config.num_layers)
    # counts[color][layer] = torch.zeros(len(colors), model.config.num_layers)



if __name__ == "__main__":
    # parse arguments for prompt template file, colors list file, output dir, model id
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--prompts_path", type=str, default="data/prompts.json")
    parser.add_argument("--objects_path", type=str, default="data/objects.json")
    parser.add_argument("--colorlist_path", type=str, default="data/colorlist.json")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    main(**args.__dict__)