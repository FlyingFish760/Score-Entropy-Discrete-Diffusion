import re
from torch import Tensor
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from itertools import chain
import numpy as np
import torch

import urllib.request
import zipfile
import requests
import json
from datasets import Dataset

from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Dataset as TorchDataset

def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string

def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x

def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n'+text.strip()


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset


def get_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=8):
    if name == "wikitext103":
        dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)
    elif name == "wikitext2":
        dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
    elif name == "ptb":
        dataset = load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif name == "lambada":
        dataset = get_lambada_test_dataset()
    else:
        dataset = load_dataset(name, cache_dir=cache_dir)

    if name == "lambada":
        data = dataset
    else:
        data = dataset[mode]

    if name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif name == "ptb":
        detokenizer = ptb_detokenizer
    elif name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif name == "lambada":
        detokenizer = lambada_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                 text[i] = detokenizer(t)
            return text
        return detok

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    EOS = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess_and_tokenize(example):
        if name == "ptb":
            text = example['sentence']
        else:
            text = example["text"]
        # print(list(example.keys()))
        # exit()
        
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokens = tokenizer(text, return_attention_mask=False)
        # add in EOS token following 
        # https://github.com/jcpeterson/openwebtext/blob/master/tokenize_text.py#L67
        for token in tokens['input_ids']:
            token.append(EOS)
        return tokens
    
    tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
    if name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns('sentence')
    else:
        tokenized_dataset = tokenized_dataset.remove_columns('text')
    

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
    chunked_dataset = chunked_dataset.with_format('torch')

    return chunked_dataset

#-------------------------- new --------------------------
def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer,
                               pad_in_loss: bool = False):
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens
    and 0 for the prompt tokens. Padding tokens (EOS) are controlled by `pad_in_loss`.

    Args:
        prompt_strs (list[str]): List of prompt strings.
        output_strs (list[str]): List of output strings.
        tokenizer (PreTrainedTokenizer): Tokenizer to use for tokenization.
        pad_in_loss (bool): Whether the EOS padding tokens are included in the loss.
            - False (default): padding mask = 0, i.e. only the real response tokens
              are supervised (prompt and padding are excluded from the loss).
            - True: padding mask = 1, i.e. the trailing EOS padding is also supervised,
              teaching the model to fill the tail with EOS (a stronger "stop" signal).
              Prompt tokens are still excluded (mask = 0).

    Returns:
        dict[str, torch.Tensor]: Let prompt_and_output_lens be a list containing the lengths of
        the tokenized prompt and output strings. The returned dictionary has:
            - token_ids (torch.Tensor of shape (batch_size, max(prompt_and_output_lens)):
              the tokenized prompt+output strings.
            - response_mask (torch.Tensor of shape (batch_size, max(prompt_and_output_lens)):
              a mask on the response tokens in the labels.
    """
    # Tokenize prompts and outputs, and construct reponse_mask
    tokenized_ids = []
    response_mask = []
    max_prompt_output_len = 0
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_id = tokenizer.encode(prompt, add_special_tokens=False)
        output_id = tokenizer.encode(output, add_special_tokens=False)
        tokenized_id = prompt_id + output_id
        tokenized_ids.append(tokenized_id)

        prompt_output_len = len(tokenized_id)
        if prompt_output_len > max_prompt_output_len:
            max_prompt_output_len = prompt_output_len

        mask = len(prompt_id) * [0] + len(output_id) * [1]
        response_mask.append(mask)

    # Pad tokenized_ids and response_mask
    
    padded_ids = []
    padded_masks = []
    EOS = tokenizer.encode(tokenizer.eos_token)[0]
    pad_mask_value = 1 if pad_in_loss else 0
    for tokenized_id, mask in zip(tokenized_ids, response_mask):
        pad_len = max_prompt_output_len - len(tokenized_id)
        padded_ids.append(tokenized_id + (pad_len * [EOS]))
        padded_masks.append(mask + (pad_len * [pad_mask_value]))

    # Construct tensors
    encoding = torch.tensor(padded_ids)
    response_mask = torch.tensor(padded_masks)
    
    res = {
        "token_ids": encoding,
        "response_mask": response_mask
    }

    return res

class SFTDataset(TorchDataset):
    def __init__(self, data_path, pad_in_loss: bool = False) -> None:
        super().__init__()

        self.pad_in_loss = pad_in_loss
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenized_data = self.tokenize_data(data_path, self.tokenizer)
        self.token_ids = tokenized_data["token_ids"]
        self.response_masks = tokenized_data["response_mask"]

    def tokenize_data(self, data_path, tokenizer)-> dict[str, Tensor]:
        prompts = []
        outputs = []
        answers = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                prompts.append(sample["prompt"])
                outputs.append(sample["response"])
                # ground-truth answer (str) for grading; "" if not present
                answers.append(str(sample.get("answer", "")))
        # keep the raw answers aligned with token_ids for eval-time grading
        self.answers = answers

        tokenized_data = tokenize_prompt_and_output(prompts, outputs, tokenizer,
                                                    pad_in_loss=self.pad_in_loss)
        return tokenized_data

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, index) -> tuple:
        # answer is a plain str; default_collate batches these into a list[str]
        return self.token_ids[index], self.response_masks[index], self.answers[index]
#-------------------------- new --------------------------
        
def get_dataloaders(config, distributed=True):
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")


    train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length)
    valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "text8" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)

    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None
    

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(test_sampler is None),
    ))
    return train_loader, valid_loader


if __name__ == "__main__":
    pass

    # prompt_strs = ["a", "b c"]
    # output_strs = ["fafsfdsfa", "1"]
    # tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    # res = tokenize_prompt_and_output(
    #     prompt_strs,
    #     output_strs,
    #     tokenizer
    # )
    # print(res)