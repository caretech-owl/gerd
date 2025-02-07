"""Data utilities for training and data processing."""

import re
from typing import Dict, Generator, List

import torch
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import PreTrainedTokenizer


def split_chunks(
    arr: List[int], size: int, step: int
) -> Generator[List[int], None, None]:
    """Splits a list of encoded tokens into chunks of a given size.

    Parameters:
        arr: The list of encoded tokens.
        size: The size of the chunks.
        step: The step size for the chunks.

    Returns:
        A generator that yields the chunks
    """
    for i in range(0, len(arr), step):
        yield arr[i : i + size]


def despacyfy(text: str) -> str:
    """Removes spacy-specific tokens from a text.

    For instance, -RRB- is replaced with ')', -LRB- with '(' and -UNK- with '*'.

    Parameters:
        text: The text to despacyfy.

    Returns:
        The despacyfied text
    """
    res = (
        text.replace("-RRB-", ")")
        .replace("-LRB-", "(")
        .replace("-UNK-", "*")
        .replace("( ", "(")
        .replace(" )", ")")
        .replace("  ", " ")
    )
    check = re.findall(r"-(RRB|UNK|LRB)-", res)
    if len(check) != 0:
        msg = f"Did not expect to find {check} in\n{res}."
        raise RuntimeError(msg)
    return res
    # check = re.findall(r'(B|I)-(PER|SALUTE)', text)
    # assert len(check) == 0, f"{check}\n{text}"


def encode(
    text: str, add_bos_token: bool, tokenizer: PreTrainedTokenizer, cutoff_len: int
) -> List[int]:
    """Encodes a text using a tokenizer.

    Parameters:
        text: The text to encode
        add_bos_token: Whether to add the beginning of sentence token
        tokenizer: The tokenizer to use
        cutoff_len: The maximum length of the encoded text

    Returns:
        The text encoded as a list of tokenizer tokens
    """
    result: list[int] = tokenizer.encode(text, truncation=True, max_length=cutoff_len)
    # Check if the first two tokens are BOS
    if len(result) >= 2 and result[:2] == [
        tokenizer.bos_token_id,
        tokenizer.bos_token_id,
    ]:
        result = result[1:]

    if not add_bos_token and result[0] == tokenizer.bos_token_id:
        result = result[1:]
    return result


def tokenize(
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    cutoff_len: int,
    append_eos_token: bool = False,
) -> Dict[str, torch.Tensor | list[int]]:
    """Converts a prompt into a tokenized input for a model.

    The methods returns the tokenized input as a dictionary with the keys
    "input_ids", "labels" and "attention_mask" where the input_ids are the
    tokenized input, the labels assign the same label ('1') to each token
    and the attention_mask masks out the padding tokens.
    Parameters:
        prompt: The prompt to tokenize
        tokenizer: The tokenizer to use
        cutoff_len: The maximum length of the encoded text
        append_eos_token: Whether to append an end of sentence token

    Returns:
        The tokenized input as a dictionary
    """
    input_ids = encode(prompt, True, tokenizer, cutoff_len)

    if tokenizer.pad_token_id is None or tokenizer.padding_side is None:
        msg = (
            "Tokenizing implies tokenizer.pad_token_id "
            "and tokenizer.padding_side to be set!"
        )
        raise AttributeError(msg)

    if (
        append_eos_token
        and input_ids[-1] != tokenizer.eos_token_id
        and len(input_ids) < cutoff_len
    ):
        input_ids.append(tokenizer.eos_token_id)

    input_ids = [tokenizer.pad_token_id] * (cutoff_len - len(input_ids)) + input_ids
    labels = [1] * len(input_ids)

    input_tensors = torch.tensor(input_ids)
    return {
        "input_ids": input_tensors,
        "labels": labels,
        "attention_mask": input_tensors.ne(tokenizer.pad_token_id),
    }
