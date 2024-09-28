import gzip
import json

import torch
import typer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm import tqdm
from transformers import LayoutLMv3ForTokenClassification

from helpers import (
    DataCollator,
    check_duplicate,
    MAX_LEN,
    parse_logits,
    prepare_inputs,
)

app = typer.Typer()

chen_cherry = SmoothingFunction()


@app.command()
def main(
    input_file: str = typer.Argument(..., help="input file"),
    model_path: str = typer.Argument(..., help="model path"),
    batch_size: int = typer.Option(16, help="batch size"),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        LayoutLMv3ForTokenClassification.from_pretrained(model_path, num_labels=MAX_LEN)
        .bfloat16()
        .to(device)
        .eval()
    )
    data_collator = DataCollator()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    datasets = []
    with gzip.open(input_file, "rt") as f:
        for line in tqdm(f):
            datasets.append(json.loads(line))
    # make batch faster
    datasets.sort(key=lambda x: len(x["source_boxes"]), reverse=True)

    total = 0
    total_out_idx = 0.0
    total_out_token = 0.0
    for i in tqdm(range(0, len(datasets), batch_size)):
        batch = datasets[i : i + batch_size]
        model_inputs = data_collator(batch)
        model_inputs = prepare_inputs(model_inputs, model)
        # forward
        with torch.no_grad():
            model_outputs = model(**model_inputs)
        logits = model_outputs.logits.cpu()
        for data, logit in zip(batch, logits):
            target_index = data["target_index"][:MAX_LEN]
            pred_index = parse_logits(logit, len(target_index))
            assert len(pred_index) == len(target_index)
            assert not check_duplicate(pred_index)
            target_texts = data["target_texts"][:MAX_LEN]
            source_texts = data["source_texts"][:MAX_LEN]
            pred_texts = []
            for idx in pred_index:
                pred_texts.append(source_texts[idx])
            total += 1
            total_out_idx += sentence_bleu(
                [target_index],
                [i + 1 for i in pred_index],
                smoothing_function=chen_cherry.method2,
            )
            total_out_token += sentence_bleu(
                [" ".join(target_texts).split()],
                " ".join(pred_texts).split(),
                smoothing_function=chen_cherry.method2,
            )

    print("total: ", total)
    print("out_idx: ", round(100 * total_out_idx / total, 1))
    print("out_token: ", round(100 * total_out_token / total, 1))


if __name__ == "__main__":
    app()
