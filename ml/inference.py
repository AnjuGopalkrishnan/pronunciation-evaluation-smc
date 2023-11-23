from io import StringIO
from speechbrain.utils.metric_stats import ErrorRateStats
import os
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import librosa
global asr_brain

class ASR(sb.Brain):
    # Testing for one single audio file
    def compute_forward_evaluate(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs = batch

        # creating a dummy wav_lens with shape of [batch,1] with 1
        wav_lens = torch.ones((1, 1)).to(self.device)

        # if stage == sb.Stage.TRAIN:
        #     if hasattr(self.hparams, "augmentation"):
        #         wavs = self.hparams.augmentation(wavs, wav_lens)

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
        ).input_values  # since its only 1 file not taking [0]

        sig = torch.Tensor(sig)
        return sig

    def get_predicted_phonemes_for_test_audio(self, test_audio_path):
        print(f"Using librosa to load the audio")
        batch = self.prepare_test_audio_for_inference(test_audio_path)
        print(f"Loading the best model & setting to eval mode")
        self.on_evaluate_start(min_key="PER")  # We call the on_evaluate_start that will load the best model
        self.modules.eval()  # We set the model to eval mode (remove dropout etc)
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
        canonical_lens = torch.ones((1, 1))

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
        alignments_end = lines.index("================================================================================",
                                     alignments_start + 1)
        alignments_lines = lines[alignments_end + 1:]

        # Process alignments
        canonical = [phn.strip() for phn in alignments_lines[1].split(';')]
        operator = [op.strip() for op in alignments_lines[2].split(';')]
        predicted = [phn.strip() for phn in alignments_lines[3].split(';')]

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

    # Training
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)

        wavs, wav_lens = batch.sig
        # phns_bos, _ = batch.phn_encoded_bos

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)
        feats = self.modules.wav2vec2(wavs)
        x = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        return p_ctc, wav_lens

# instantiate only once somewhere ( wave2vec2_model_init.py)
hparams_file = '/Users/dhanushlalitha/Documents/SEM2/SOUND_AND_MUSIC/pronunciation-evaluation-smc/ml/config/wave2vec2/hparams/train.yaml'

# Load hyperparameters file with command-line overrides
with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin)

def initiate_asr():
    #use the global variable asr_brain
    global asr_brain
    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = sb.dataio.encoder.CTCTextEncoder()
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "blank_label": hparams["blank_index"],
    }
    asr_brain.label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[],  # Provide the datasets if needed
        output_key="phn_list_target",
        special_labels=special_labels,
        sequence_input=True,
    )
    return asr_brain
