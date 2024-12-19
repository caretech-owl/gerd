import re
from typing import Dict, Generator, List

import torch
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import PreTrainedTokenizer


def split_chunks(
    arr: List[int], size: int, step: int
) -> Generator[List[int], None, None]:
    for i in range(0, len(arr), step):
        yield arr[i : i + size]


def despacyfy(text: str) -> str:
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
    result = tokenizer.encode(text, truncation=True, max_length=cutoff_len)
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
) -> Dict[str, List]:
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

    input_ids = torch.tensor(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": input_ids.ne(tokenizer.pad_token_id),
    }


def generate_and_tokenize_prompt(
    data_point: Dict[str, str],
    format_template: Dict[str, str],
    tokenizer: PreTrainedTokenizer,
    cutoff_len: int,
    append_eos_token: bool = False,
) -> Dict[str, List]:
    for options, prompt in format_template.items():
        if set(options.split(",")) == {
            x[0]
            for x in data_point.items()
            if (isinstance(x[1], str) and len(x[1].strip()) > 0)
        }:
            for key, val in data_point.items():
                if isinstance(val, str):
                    prompt = prompt.replace(f"%{key}%", val)
            return tokenize(
                prompt,
                tokenizer,
                cutoff_len=cutoff_len,
                append_eos_token=append_eos_token,
            )
    msg = (
        f'Data-point "{data_point}" has no keyset match '
        'within format "{list(format_data.keys())}"'
    )
    raise RuntimeError(msg)


def split_plain_text(txt: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=25)
    res = []
    txts = splitter.split_text(txt)
    for substring in txts:
        pos = txt.index(substring)
        col = 0
        line = 0
        for c in txt[:pos]:
            if c == "\n":
                line += 1
                col = 0
            else:
                col += 1
        res.append(
            Document(page_content=substring, metadata={"position": f"{line}:{col}"})
        )
    return res
