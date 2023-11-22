from io import StringIO
from speechbrain.utils.metric_stats import ErrorRateStats
import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import librosa
import csv

# This function will come in a file called wave2vec_sb.py
def make_attn_mask(wavs, wav_lens):
    """
    wav_lens: relative lengths(i.e. 0-1) of a batch. shape: (bs, )
    return a tensor of shape (bs, seq_len), representing mask on allowed positions.
            1 for regular tokens, 0 for padded tokens
    """
    abs_lens = (wav_lens*wavs.shape[1]).long()
    attn_mask = wavs.new(wavs.shape).zero_().long()
    for i in range(len(abs_lens)):
        attn_mask[i, :abs_lens[i]] = 1
    return attn_mask

# This class will come in a file called wave2vec_sb.py
class ASR(sb.Brain):
    #Testing for one single audio file
    def compute_forward_evaluate(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs = batch

        #creating a dummy wav_lens with shape of [batch,1] with 1
        wav_lens = torch.ones((1,1)).to(self.device)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        attn_mask = None
        feats = self.modules.wav2vec2(wavs)

        x = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)
        # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
        # that is, it return a list of list with different lengths
        sequence = sb.decoders.ctc_greedy_decode(
            p_ctc, wav_lens, blank_id=self.hparams.blank_index
        )
        transcriptions = [" ".join(self.label_encoder.decode_ndim(s)) for s in sequence]
        return transcriptions, sequence

    def prepare_test_audio_for_inference(self, test_audio_path):
        # Use wav2vec processor to do normalization
        audio_signal, _ = librosa.core.load(test_audio_path, sr=self.hparams.sample_rate)
        sig = self.hparams.wav2vec2.feature_extractor(
            audio_signal,
            sampling_rate=self.hparams.sample_rate,
        ).input_values #since its only 1 file not taking [0]

        sig = torch.Tensor(sig)
        return sig

    def get_predicted_phonemes_for_test_audio(self, test_audio_path):
        print(f"Using librosa to load the audio")
        batch = self.prepare_test_audio_for_inference(test_audio_path)
        print(f"Loading the best model & setting to eval mode")
        self.on_evaluate_start(min_key="PER") # We call the on_evaluate_start that will load the best model
        self.modules.eval() # We set the model to eval mode (remove dropout etc)
        self.modules.wav2vec2.model.config.apply_spec_augment = False  # make sure no spec aug applied on wav2vec2

        with torch.no_grad():
          print("Making predictions from the best model")
          preds, seq = self.compute_forward_evaluate(batch, stage=sb.Stage.TEST)
          print("Got the predictions")
        return preds, seq

    def evaluate_test_audio(self, test_audio_path, canonical_phonemes):
        predicted_phonemes, predicted_sequence = self.get_predicted_phonemes_for_test_audio(test_audio_path)

        print("Converting canonical to appropriate format for getting error")
        phn_list_canonical = canonical_phonemes.strip().split()
        phn_encoded_list_canonical = [self.label_encoder.encode_sequence(phn_list_canonical)]
        canonicals = torch.LongTensor(phn_encoded_list_canonical)
        canonical_lens = torch.ones((1,1))

        print("Getting the error stats")
        error_metrics = ErrorRateStats()
        error_metrics.append(
                        ids=[test_audio_path],
                        predict=predicted_sequence,
                        target=canonicals,
                        predict_len=None,
                        target_len=canonical_lens,
                        ind2lab=self.label_encoder.decode_ndim,
                    )
        stats = error_metrics.summarize()
        # get score (100 - WER)
        score = 100 - stats["WER"]
        print(f"Calculated the score to be: {score}")
        print("Now capturing the stats sysout in a variable")
        # get the errors
        # Redirect sys.stdout to capture the output
        original_stdout = sys.stdout
        sys.stdout = StringIO()

        # Call write_stats
        error_metrics.write_stats(None)

        # Get the content of the buffer
        stats_string = sys.stdout.getvalue()

        # Reset sys.stdout
        sys.stdout = original_stdout
        print("Extracting stats from stdout")
        return predicted_phonemes, score, self.extract_stats_from_wer_stats_string(stats_string)

    def extract_stats_from_wer_stats_string(self, stats_string):
        lines = stats_string.split('\n')
        lines = [line.strip() for line in lines]

        # Find the start and end of the ALIGNMENTS section
        alignments_start = lines.index("ALIGNMENTS")
        alignments_end = lines.index("================================================================================", alignments_start+1)
        alignments_lines = lines[alignments_end+1:]

        # Process alignments
        canonical = [phn.strip() for phn in alignments_lines[1].split(';')]
        operator = [op.strip() for op in alignments_lines[2].split(';')]
        predicted = [phn.strip() for phn in  alignments_lines[3].split(';')]

        # Initialize error categories
        errors = {
            "deletions": {"canonical": [], "predicted": []},
            "insertions": {"canonical": [], "predicted": []},
            "substitutions": {"canonical": [], "predicted": []},
            "canonical": canonical,
            "predicted": predicted
        }

        for i, item in enumerate(zip(canonical, operator, predicted)):
            canonical_phn, op, predicted_phn = item
            if op == "I":
                errors["insertions"]["canonical"].append((i, canonical_phn))
                errors["insertions"]["predicted"].append((i, predicted_phn))
            elif op == "S":
                errors["substitutions"]["canonical"].append((i, canonical_phn))
                errors["substitutions"]["predicted"].append((i, predicted_phn))
            elif op == "D":
                errors["deletions"]["canonical"].append((i, canonical_phn))
                errors["deletions"]["predicted"].append((i, predicted_phn))

        return errors

    #Training
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)

        wavs, wav_lens = batch.sig
        # phns_bos, _ = batch.phn_encoded_bos

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # some wav2vec models (e.g. large-lv60) needs attention_mask
        if self.modules.wav2vec2.feature_extractor.return_attention_mask:
            attn_mask = make_attn_mask(wavs, wav_lens)
            feats = self.modules.wav2vec2(wavs, attention_mask=attn_mask)
        else:
            attn_mask = None
            feats = self.modules.wav2vec2(wavs)

        x = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."

        p_ctc, wav_lens = predictions

        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        if stage != sb.Stage.TRAIN:
            canonicals, canonical_lens = batch.phn_encoded_canonical
            perceiveds, perceived_lens = batch.phn_encoded_perceived

        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss = loss_ctc

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)

            self.per_metrics.append(
                ids=ids,
                predict=sequence,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            self.mpd_metrics.append(
                ids=ids,
                predict=sequence,
                canonical=canonicals,
                perceived=perceiveds,
                predict_len=None,
                canonical_len=canonical_lens,
                perceived_len=perceived_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )

        return loss

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()
        if self.hparams.wav2vec2_specaug:
            self.modules.wav2vec2.model.config.apply_spec_augment = True

        if stage != sb.Stage.TRAIN:
            self.modules.wav2vec2.model.config.apply_spec_augment = False
            self.per_metrics = self.hparams.per_stats()
            self.mpd_metrics = MpdStats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")
            mpd_f1 = self.mpd_metrics.summarize("mpd_f1")

        if stage == sb.Stage.VALID:

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_adam": self.adam_optimizer.param_groups[0]["lr"],
                    "lr_wav2vec": self.wav2vec_optimizer.param_groups[0]["lr"],
                },
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "loss": stage_loss,
                    "ctc_loss": self.ctc_metrics.summarize("average"),
                    "PER": per,
                    "mpd_f1": mpd_f1
                },
            )
            self.checkpointer.save_and_keep_only(
                meta={"PER": per, "mpd_f1": mpd_f1}, min_keys=["PER"], max_keys=["mpd_f1"]
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per, "mpd_f1": mpd_f1},
            )
            with open(self.hparams.wer_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print(
                    "CTC and PER stats written to file",
                    self.hparams.wer_file,
                )
            with open(self.hparams.mpd_file, "w") as m:
                m.write("MPD results and stats:\n")
                self.mpd_metrics.write_stats(m)
                print(
                    "MPD results and stats written to file",
                    self.hparams.mpd_file,
                )

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        # Managing automatic mixed precision
        if self.auto_mix_prec:

            self.wav2vec_optimizer.zero_grad()
            self.adam_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.wav2vec_optimizer)
            self.scaler.unscale_(self.adam_optimizer)

            if self.check_gradients(loss):
                self.scaler.step(self.wav2vec_optimizer)
                self.scaler.step(self.adam_optimizer)

            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            # normalize the loss by gradient_accumulation step
            (loss / self.hparams.gradient_accumulation).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                # gradient clipping & early stop if loss is not fini
                if self.check_gradients(loss):
                    self.wav2vec_optimizer.step()
                    self.adam_optimizer.step()

                self.wav2vec_optimizer.zero_grad()
                self.adam_optimizer.zero_grad()

        return loss.detach().cpu()

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.model.parameters()
        )
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile_jit()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        ## NOTE: make sure to use the "best" model to continual training
        ## so we set the `min_key` argument
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                device=torch.device(self.device),
                min_key="PER"
            )

# This class will come in a file called wave2vec_sb.py
def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder_save"]
    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        # sig = sb.dataio.dataio.read_audio(wav)
        # # sample rate change to 16000, e,g, using librosa
        # sig = torch.Tensor(librosa.core.load(wav, hparams["sample_rate"])[0])
        # Use wav2vec processor to do normalization
        sig = hparams["wav2vec2"].feature_extractor(
            librosa.core.load(wav, sr=hparams["sample_rate"])[0],
            sampling_rate=hparams["sample_rate"],
        ).input_values[0]
        sig = torch.Tensor(sig)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("perceived_train_target")
    @sb.utils.data_pipeline.provides(
        "phn_list_target",
        "phn_encoded_list_target",
        "phn_encoded_target",
    )
    def text_pipeline_train(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded_list = label_encoder.encode_sequence(phn_list)
        yield phn_encoded_list
        phn_encoded = torch.LongTensor(phn_encoded_list)
        yield phn_encoded

    @sb.utils.data_pipeline.takes("perceived_train_target", "canonical_aligned", "perceived_aligned")
    @sb.utils.data_pipeline.provides(
        "phn_list_target",
        "phn_encoded_list_target",
        "phn_encoded_target",
        "phn_list_canonical",
        "phn_encoded_list_canonical",
        "phn_encoded_canonical",
        "phn_list_perceived",
        "phn_encoded_list_perceived",
        "phn_encoded_perceived",
    )
    def text_pipeline_test(target, canonical, perceived):
        phn_list_target = target.strip().split()
        yield phn_list_target
        phn_encoded_list_target = label_encoder.encode_sequence(phn_list_target)
        yield phn_encoded_list_target
        phn_encoded_target = torch.LongTensor(phn_encoded_list_target)
        yield phn_encoded_target
        phn_list_canonical = canonical.strip().split()
        yield phn_list_canonical
        phn_encoded_list_canonical = label_encoder.encode_sequence(phn_list_canonical)
        yield phn_encoded_list_canonical
        phn_encoded_canonical = torch.LongTensor(phn_encoded_list_canonical)
        yield phn_encoded_canonical
        phn_list_perceived = perceived.strip().split()
        yield phn_list_perceived
        phn_encoded_list_perceived = label_encoder.encode_sequence(phn_list_perceived)
        yield phn_encoded_list_perceived
        phn_encoded_perceived = torch.LongTensor(phn_encoded_list_perceived)
        yield phn_encoded_perceived

    sb.dataio.dataset.add_dynamic_item([train_data], text_pipeline_train)
    sb.dataio.dataset.add_dynamic_item([valid_data, test_data], text_pipeline_test)

    # 3. Fit encoder:
    # Load or compute the label encoder
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "blank_label": hparams["blank_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="phn_list_target",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        [train_data],
        ["id", "sig", "phn_encoded_target"],
    )
    sb.dataio.dataset.set_output_keys(
        [valid_data, test_data],
        ["id", "sig", "phn_encoded_target", "phn_encoded_canonical", "phn_encoded_perceived"],
    )

    return train_data, valid_data, test_data, label_encoder

# results ( which is the wave2vec2 model checkpoints will be in some separte folder) + make sure /ml/config/wave2vec2/hparams/train.yaml is in a place which can reference this

# instantiate only once somewhere ( wave2vec2_model_init.py)
hparams_file = '/content/drive/MyDrive/CS5647_Project/wave2vec2/hparams/train.yaml'

# Load hyperparameters file with command-line overrides
with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin)


# Create experiment directory
sb.create_experiment_directory(
    experiment_directory=hparams["output_folder"],
    hyperparams_to_save=hparams_file,
)

# Dataset IO prep: creating Dataset objects and proper encodings for phones
train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)

# Trainer initialization
asr_brain = ASR(
    modules=hparams["modules"],
    hparams=hparams,
    checkpointer=hparams["checkpointer"],
    run_opts={"device": "cuda"}
)
asr_brain.label_encoder = label_encoder


# this function call will come in  your backend api

'''
import asr_brain from wave2vec2_model_init

@get
def api():
    #save the audio file from the browser in a wav file and pass that path here
    
    if model_tupe== wave2vec2
        predicted_phonemes, score, stats = asr_brain.evaluate_test_audio(test_audio_path, canonical_phonemes)
    elif:
        --
    
    #based on substitutions, insertions & deletions what other words/sentences can be practiced...
    return something which the ui can use

'''

test_audio_path = "/content/drive/MyDrive/CS5647_Project/dataset/TNI/wav/arctic_a0100.wav"
canonical_phonemes = "sil y uw m ah s t s l iy p sil hh iy er jh d sil sil" # actual sentence is 'You must sleep he urged'
predicted_phonemes, score, stats = asr_brain.evaluate_test_audio(test_audio_path, canonical_phonemes)