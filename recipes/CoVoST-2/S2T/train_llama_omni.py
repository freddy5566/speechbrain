import json

import torch
import speechbrain as sb

from hyperpyyaml import load_hyperpyyaml

from omni_speech.model.builder import load_pretrained_model
from omni_speech.datasets.preprocess import tokenizer_speech_token
from omni_speech.utils import disable_torch_init

# data preparation
class SpeechToTextDataset(sb.dataio.dataset.DynamicItemDataset):
    def __init__(self, json_file, tokenizer, model_config):
        super().__init__()

        self.tokenizer = tokenizer
        self.model_config = model_config
        self.data = json.load(open(json_file, "r"))

    def __getitem__(self, index):
        item = self.data[index]
        speech_file = item["speech"]
        qs = item["conversations"][0]["value"]
        answer = item["conversations"][1]["value"]

        speech = sb.dataio.dataio.load_audio(speech_file)
        input_ids = tokenizer_speech_token(qs, self.tokenizer, return_tensors="pt")
        target_ids = tokenizer_speech_token(answer, self.tokenizer, return_tensors="pt")

        return {
            "input_ids": input_ids,
            "speech": speech,
            "target_ids": target_ids
        }

    def __len__(self):
        return len(self.data)

class ST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        input_ids, speech, target_ids = batch.input_ids, batch.speech, batch.target_ids
        input_ids, speech, target_ids = input_ids.to(self.device), speech.to(self.device), target_ids.to(self.device)

        outputs = self.modules.model.generate(
            input_ids,
            speech=speech,
            do_sample=False,
            max_new_tokens=self.hparams.max_new_tokens
        )

        return outputs, target_ids

    def compute_objectives(self, predictions, batch, stage):
        outputs, target_ids = predictions
        loss = self.hparams.compute_loss(outputs, target_ids)
        return loss


if __name__ == "__main__":
    yaml_path = "config.yaml"
    with open(yaml_path, "r", encoding="utf-8") as config_file:
        hparams = load_hyperpyyaml(config_file)

    disable_torch_init()
    tokenizer, model, _ = load_pretrained_model(hparams["model_path"], hparams["model_base"])

    # datasets
    train_data = SpeechToTextDataset(hparams["dataio_pipeline"].json_train, tokenizer, model.config)
    valid_data = SpeechToTextDataset(hparams["dataio_pipeline"].json_valid, tokenizer, model.config)

    # setup trainer
    speech_to_text_brain = ST(
        modules={"model": model},
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        checkpointer=hparams["checkpointer"]
    )

    # start trainining
    speech_to_text_brain.fit(
        epoch_counter=hparams["num_epochs"],
        train_set=train_data,
        valid_set=valid_data,
        train_loader_kwargs={"batch_size": hparams["batch_size"]},
        valid_loader_kwargs={"batch_size": hparams["batch_size"]}
    )
