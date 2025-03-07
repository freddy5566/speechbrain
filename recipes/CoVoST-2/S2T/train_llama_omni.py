import json
import os
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from omni_speech.model.builder import load_pretrained_model
import sys

def dataio_prepare(hparams):
    """
    Prepares the dataset for use with SpeechBrain's Brain class.

    Assumes that the following JSON files exist in hparams["data_folder"]:
      - train.json
      - dev.json
      - test.json

    Each JSON file should have the following structure:
      {
         "id": ...,           // Optional; SpeechBrain will generate it automatically if not present.
         "speech": "/path/to/audio/file.mp3",
         "duration": <float>, // (Optional) used for sorting
         "conversations": [
              {"user": "Please translate the given German into English."},
              {"assistent": "Neologisms are a phenomenon of lively languages."}
         ]
      }

    This function generates:
      - "sig": The audio signal loaded from "speech"
      - "user_text": The user's prompt extracted from the first conversation entry
      - "assistent_text": The assistant's response extracted from the second conversation entry
      - "user_tokens": Tokenized sequence of user_text (for model input)
      - "assistant_tokens": Tokenized sequence of assistent_text (as target labels)
    """
    def load_and_fix_json(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Convert list format to dictionary and remove "id" if present (SpeechBrain will generate id)
        if isinstance(data, list):
            new_data = {}
            for i, item in enumerate(data):
                key = item.get("id", str(i))
                if "id" in item:
                    del item["id"]
                new_data[key] = item
            data = new_data
        return data

    datasets = {}
    data_folder = hparams["data_folder"]
    file_dict = {
        "train": os.path.join(data_folder, "train.json"),
        "valid": os.path.join(data_folder, "dev.json"),
        "test": os.path.join(data_folder, "test.json"),
    }

    # Audio processing pipeline: read "speech" and output "sig"
    @sb.utils.data_pipeline.takes("speech")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(speech):
        sig = sb.dataio.dataio.read_audio(speech)
        return sig

    # Text processing pipeline:
    # Extract user_text and assistent_text from "conversations", then tokenize both.
    @sb.utils.data_pipeline.takes("conversations")
    @sb.utils.data_pipeline.provides("user_text", "assistent_text", "user_tokens", "assistant_tokens")
    def one_reference_text_pipeline(conversations):
        user_text = conversations[0]["user"]
        assistent_text = conversations[1]["assistent"]
        yield user_text                # "user_text"
        yield assistent_text           # "assistent_text"
        user_tokens = hparams["tokenizer"].encode(user_text, add_special_tokens=False)
        yield torch.LongTensor(user_tokens)   # "user_tokens"
        assistant_tokens = hparams["tokenizer"].encode(assistent_text, add_special_tokens=False)
        yield torch.LongTensor(assistant_tokens)  # "assistant_tokens"

    # Create datasets for each split
    for key, json_path in file_dict.items():
        fixed_data = load_and_fix_json(json_path)
        datasets[key] = sb.dataio.dataset.DynamicItemDataset(
            fixed_data,
            dynamic_items=[],
            output_keys=["id", "speech", "conversations"]
        )
        # Add dynamic pipelines
        datasets[key].add_dynamic_item(audio_pipeline)
        datasets[key].add_dynamic_item(one_reference_text_pipeline)
        datasets[key].set_output_keys([
            "id",
            "sig",
            "user_text",
            "assistent_text",
            "user_tokens",
            "assistant_tokens",
        ])

    # Data sorting based on hparams["sorting"]
    if hparams["sorting"] == "ascending":
        datasets["train"] = datasets["train"].filtered_sorted(sort_key="duration")
        datasets["valid"] = datasets["valid"].filtered_sorted(sort_key="duration")
        hparams["train_dataloader_opts"]["shuffle"] = False
        hparams["valid_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        datasets["train"] = datasets["train"].filtered_sorted(sort_key="duration", reverse=True)
        datasets["valid"] = datasets["valid"].filtered_sorted(sort_key="duration", reverse=True)
        hparams["train_dataloader_opts"]["shuffle"] = False
        hparams["valid_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "random":
        hparams["train_dataloader_opts"]["shuffle"] = True
    else:
        raise NotImplementedError("sorting must be random, ascending or descending")

    return datasets

class ST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        input_ids = batch.user_tokens       
        labels = batch.assistant_tokens     
        speech = batch.sig                  

        prediction = self.modules.model(
            input_ids=input_ids,
            speech=speech,
            labels=labels
        )
        return prediction
    
    def compute_objectives(self, prediction, batch, stage):
        if hasattr(prediction, "loss") and prediction.loss is not None:
            loss = prediction.loss
        else:
            loss = self.hparams["compute_loss"](prediction, batch)
        return loss    
    
    def check_and_reset_optimizer(self):
        current_epoch = self.hparams.epoch_counter.current
        if not hasattr(self, "switched"):
            self.switched = False
        if self.switched:
            return
        if current_epoch > self.hparams.stage_one_epochs:
            self.optimizer = self.hparams.optimizer_sgd(self.modules.model.parameters())
            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)
            self.switched = True

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        if should_step:
            self.hparams["scheduler"](self.optimizer)  
    
    def on_fit_batch_start(self, batch, should_step):
        self.check_and_reset_optimizer()

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.bleu_metric = self.hparams.bleu_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}
        self.hparams.train_logger.log_stats(
            stats_meta={"epoch": epoch},
            train_stats=self.train_stats,
            valid_stats=stage_stats,
        )
        self.checkpointer.save_and_keep_only(
            meta={"loss": stage_stats["loss"], "epoch": epoch},
            num_to_keep=self.hparams.avg_checkpoints
        )


    def on_fit_start(self):
        super().on_fit_start()
        current_epoch = self.hparams.epoch_counter.current
        if current_epoch > self.hparams.stage_one_epochs:
            self.optimizer = self.hparams.optimizer_sgd(self.modules.model.parameters())
            if self.checkpointer is not None:
                self.checkpointer.recover_if_possible()

    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start()
        ckpts = self.checkpointer.find_checkpoints(max_key=max_key, min_key=min_key)
        ckpt = sb.utils.checkpoints.average_checkpoints(ckpts, recoverable_name="model")
        self.modules.model.load_state_dict(ckpt, strict=True)
        self.modules.model.eval()



if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    tokenizer, model, context_len = load_pretrained_model(
        model_path=hparams["model_path"],
        model_base=None,  
        is_lora=False,
        s2s=False,
    )
    hparams["tokenizer"] = tokenizer
    hparams.setdefault("modules", {})["model"] = model
    
    datasets = dataio_prepare(hparams)
    # setup trainer
    st_brain = ST(hparams, run_opts, checkpointer = hparams["checkpointer"] )

    # start trainining
    st_brain.fit(
        st_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )
