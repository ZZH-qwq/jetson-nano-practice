# coding=utf-8
# Ref: https://github.com/xiabingquan/Automatic-Speech-Recognition-from-Scratch

import os
import sys
from dataclasses import dataclass, field

import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer import CharTokenizer, SubwordTokenizer
from dataloader import get_dataloader
from models import Encoder, Decoder, Transformer
from feature_extractors import LinearFeatureExtractionModel, ResNet1D


def init_model(vocab_size, enc_dim, num_enc_layers, num_dec_layers, feature_extractor_type):
    fbank_dim = 80
    num_heads = enc_dim // 64
    max_seq_len = 2048

    if feature_extractor_type == "linear":
        FeatureExtractor = LinearFeatureExtractionModel
    elif feature_extractor_type == "resnet":
        FeatureExtractor = ResNet1D
    else:
        raise ValueError(f"Unsupported feature extractor type: {feature_extractor_type}")
    feature_extractor = FeatureExtractor(fbank_dim, enc_dim)

    encoder = Encoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=num_enc_layers, enc_dim=enc_dim, num_heads=num_heads, dff=2048, tgt_len=max_seq_len
    )

    decoder = Decoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=num_dec_layers, dec_dim=enc_dim, num_heads=num_heads, dff=2048, tgt_len=max_seq_len,
        tgt_vocab_size=vocab_size
    )

    model = Transformer(feature_extractor, encoder,
                        decoder, enc_dim, vocab_size)

    return model


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print(
            "Usage: python train.py <feature_extractor_type> <dataset_type> <ckpt_path>")
        sys.exit(1)
    feature_extractor_type = sys.argv[1]
    dataset_type = sys.argv[2]
    ckpt_path = sys.argv[3]
    assert feature_extractor_type in ["linear", "resnet"]
    assert dataset_type in ["lrs2", "librispeech-100",
                            "librispeech-360", "librispeech-500"]

    if dataset_type == "lrs2":
        t_ph = "./spm/lrs2/1000_bpe.model"  # path to the tokenizer
        audio_path_file = "./data/LRS2/train.paths"
        text_file = "./data/LRS2/train.text"
        lengths_file = "./data/LRS2/train.lengths"
    elif dataset_type == "librispeech-100":
        t_ph = "./spm/librispeech-500/1000_bpe.model"
        audio_path_file = "./data/LibriSpeech/train-clean-100.paths"
        text_file = "./data/LibriSpeech/train-clean-100.text"
        lengths_file = "./data/LibriSpeech/train-clean-100.lengths"
    elif dataset_type == "librispeech-360":
        t_ph = "./spm/librispeech-500/1000_bpe.model"
        audio_path_file = "./data/LibriSpeech/train-clean-360.paths"
        text_file = "./data/LibriSpeech/train-clean-360.text"
        lengths_file = "./data/LibriSpeech/train-clean-360.lengths"
    elif dataset_type == "librispeech-500":
        t_ph = "./spm/librispeech-500/1000_bpe.model"
        audio_path_file = "./data/LibriSpeech/train-other-500.paths"
        text_file = "./data/LibriSpeech/train-other-500.text"
        lengths_file = "./data/LibriSpeech/train-other-500.lengths"
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # define tokenizer
    tokenizer = SubwordTokenizer(t_ph)
    print(tokenizer)

    # load data
    with open(audio_path_file, 'r') as f:
        audio_paths = f.read().splitlines()
    with open(text_file, 'r') as f:
        transcripts = f.read().splitlines()
    with open(lengths_file, 'r') as f:
        wav_lengths = f.read().splitlines()
    wav_lengths = [float(length) for length in wav_lengths]

    # create checkpoint directory
    ckpt_dir = f"./.checkpoints_{feature_extractor_type}_{dataset_type}_2"
    os.makedirs(ckpt_dir, exist_ok=True)

    # define dataloader
    batch_size = 64
    batch_seconds = 512         # depends on your GPU memory
    data_loader = get_dataloader(
        audio_paths, transcripts, wav_lengths, tokenizer, batch_size, batch_seconds, shuffle=True
    )

    # define model
    vocab = tokenizer.vocab
    enc_dim = 256
    num_enc_layers = 12
    num_dec_layers = 6
    model = init_model(vocab, enc_dim, num_enc_layers,
                       num_dec_layers, feature_extractor_type)
    print(model)
    model.train()

    # load checkpoint
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))

    # DataParallel for multi-gpu
    if torch.cuda.device_count() > 1:
        dp = True
        model = nn.DataParallel(model)
    else:
        dp = False
    if torch.cuda.is_available():
        model.cuda()

    # define optimizer and scheduler
    max_lr = 4e-4
    num_epoch = 50
    num_warmup = 10000

    # percentage of warmup
    pcb = num_warmup / (len(data_loader) * num_epoch)
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, steps_per_epoch=len(data_loader), epochs=num_epoch,
        pct_start=pcb, anneal_strategy="cos",
    )

    # define loss criterion
    criterion = nn.CrossEntropyLoss(
        ignore_index=-1, label_smoothing=0.1)        # -1: ignore padding

    # main loop
    pbar = tqdm.tqdm(range(len(data_loader)), desc="Training")
    for epoch in range(1, num_epoch + 1):

        tot_loss = 0.
        data_loader.dataset.shuffle()

        for i, batch in enumerate(data_loader, start=1):

            # get batch data
            fbank_feat, feat_lens, ys_in_pad, ys_out_pad = batch
            if torch.cuda.is_available():
                fbank_feat = fbank_feat.cuda()
                feat_lens = feat_lens.cuda()
                ys_in_pad = ys_in_pad.cuda()
                ys_out_pad = ys_out_pad.cuda()

            # forward
            logits = model(fbank_feat, feat_lens, ys_in_pad)

            # calculate loss
            logits = logits.view(-1, logits.size(-1))
            ys_out_pad = ys_out_pad.view(-1).long()
            loss = criterion(logits, ys_out_pad)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # refresh progress bar
            tot_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{tot_loss / i:.2f}",
                "epoch": f"{epoch}/{num_epoch}",
            })
            pbar.update(1)
        pbar.reset()

        print(
            f"Epoch: {epoch:02d}/{num_epoch:02d}, Loss: {tot_loss / len(data_loader):.2f}")

        # save model
        torch.save(
            model.module.state_dict() if dp else model.state_dict(),
            os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
        )
