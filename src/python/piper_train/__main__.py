import argparse
import json
import logging
from pathlib import Path
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from delay_save_checkpoint import DelayedModelCheckpoint

from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, help="Path to pre-processed dataset directory"
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        help="Save checkpoint every N epochs (default: 1)",
    )
    parser.add_argument(
        "--quality",
        default="medium",
        choices=("x-low", "medium-low", "medium", "high"),
        help="Quality/size of model (default: medium)",
    )
    parser.add_argument(
        "--resume_from_single_speaker_checkpoint",
        help="For multi-speaker models only. Converts a single-speaker checkpoint to multi-speaker and resumes training",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop training after N epochs without val_loss improvement (disabled by default)",
    )
    Trainer.add_argparse_args(parser)
    VitsModel.add_model_specific_args(parser)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    _LOGGER.debug(args)

    args.dataset_dir = Path(args.dataset_dir)
    if not args.default_root_dir:
        args.default_root_dir = args.dataset_dir

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    config_path = args.dataset_dir / "config.json"
    dataset_path = args.dataset_dir / "dataset.jsonl"

    with open(config_path, "r", encoding="utf-8") as config_file:
        # See preprocess.py for format
        config = json.load(config_file)
        num_symbols = int(config["num_symbols"])
        num_speakers = int(config["num_speakers"])
        sample_rate = int(config["audio"]["sample_rate"])

    # Phonemize custom test sentences
    test_sentences = None
    if args.test_sentences is not None:
        from piper_phonemize import phonemize_espeak, phoneme_ids_espeak

        language = config["espeak"]["voice"]
        test_sentences = []
        with open(args.test_sentences, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                phonemes = [
                    p for sentence_phonemes in phonemize_espeak(line, language)
                    for p in sentence_phonemes
                ]
                ids = phoneme_ids_espeak(phonemes)
                test_sentences.append({"text": line, "phoneme_ids": ids})
        _LOGGER.info(
            "Phonemized %s test sentence(s) from %s",
            len(test_sentences), args.test_sentences,
        )

    callbacks = []
    if args.checkpoint_epochs is not None:
        callbacks.append(ModelCheckpoint(every_n_epochs=args.checkpoint_epochs))
        _LOGGER.debug(
            "Checkpoints will be saved every %s epoch(s)", args.checkpoint_epochs
        )

    # Save best ckpt
    best_ckpt_dir = args.dataset_dir / "best_ckpt"
    if not os.path.exists(best_ckpt_dir):
        os.makedirs(best_ckpt_dir)

    callbacks.append(DelayedModelCheckpoint(
        start_after_step=10000,
        every_n_train_steps=1000,
        monitor="val_loss",
        mode="min",
        save_top_k=6,
        dirpath=best_ckpt_dir,
        filename='best_ckpt_{epoch:02d}_{step:06d}_{val_loss:.2f}'
    ))

    if args.early_stopping_patience is not None:
        callbacks.append(EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            mode="min",
        ))
        _LOGGER.debug(
            "Early stopping enabled with patience=%s epochs", args.early_stopping_patience
        )

    trainer = Trainer.from_argparse_args(args, callbacks=callbacks)

    dict_args = vars(args)
    dict_args["test_sentences"] = test_sentences

    # Custom validation dataset
    if args.validation_dataset_dir is not None:
        val_dataset_dir = Path(args.validation_dataset_dir)
        val_dataset_path = val_dataset_dir / "dataset.jsonl"
        dict_args["validation_dataset"] = [val_dataset_path]
        _LOGGER.info("Using custom validation dataset from %s", val_dataset_dir)
    else:
        dict_args["validation_dataset"] = None
    if args.quality == "x-low":
        dict_args["hidden_channels"] = 96
        dict_args["inter_channels"] = 96
        dict_args["filter_channels"] = 384
    elif args.quality == "medium-low":
        # Balanced configuration between x-low and medium
        # Optimized for mobile inference while maintaining good quality
        dict_args["hidden_channels"] = 128
        dict_args["inter_channels"] = 128
        dict_args["filter_channels"] = 512
        dict_args["upsample_initial_channel"] = 192
    elif args.quality == "high":
        dict_args["resblock"] = "1"
        dict_args["resblock_kernel_sizes"] = (3, 7, 11)
        dict_args["resblock_dilation_sizes"] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        )
        dict_args["upsample_rates"] = (8, 8, 2, 2)
        dict_args["upsample_initial_channel"] = 512
        dict_args["upsample_kernel_sizes"] = (16, 16, 4, 4)

    model = VitsModel(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        dataset=[dataset_path],
        **dict_args,
    )

    if args.resume_from_single_speaker_checkpoint:
        assert (
            num_speakers > 1
        ), "--resume_from_single_speaker_checkpoint is only for multi-speaker models. Use --resume_from_checkpoint for single-speaker models."

        # Load single-speaker checkpoint
        _LOGGER.debug(
            "Resuming from single-speaker checkpoint: %s",
            args.resume_from_single_speaker_checkpoint,
        )
        model_single = VitsModel.load_from_checkpoint(
            args.resume_from_single_speaker_checkpoint,
            dataset=None,
        )
        g_dict = model_single.model_g.state_dict()
        for key in list(g_dict.keys()):
            # Remove keys that can't be copied over due to missing speaker embedding
            if (
                key.startswith("dec.cond")
                or key.startswith("dp.cond")
                or ("enc.cond_layer" in key)
            ):
                g_dict.pop(key, None)

        # Copy over the multi-speaker model, excluding keys related to the
        # speaker embedding (which is missing from the single-speaker model).
        load_state_dict(model.model_g, g_dict)
        load_state_dict(model.model_d, model_single.model_d.state_dict())
        _LOGGER.info(
            "Successfully converted single-speaker checkpoint to multi-speaker"
        )

    trainer.fit(model)


def load_state_dict(model, saved_state_dict):
    state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in saved_state_dict:
            # Use saved value
            new_state_dict[k] = saved_state_dict[k]
        else:
            # Use initialized value
            _LOGGER.debug("%s is not in the checkpoint", k)
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
