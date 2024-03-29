from io import StringIO

import librosa
from speechbrain.dataio.encoder import CTCTextEncoder
from speechbrain.utils.metric_stats import ErrorRateStats
import speechbrain as sb
import torch
import sys
from hyperpyyaml import load_hyperpyyaml
import os

# Util function needed for the class to work
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
          print(f"Got the predictions which is {preds}")
        return preds[0], seq

    def evaluate_test_audio(self, test_audio_path, canonical_phonemes):
        predicted_phonemes, predicted_sequence = self.get_predicted_phonemes_for_test_audio(test_audio_path)
        predicted_phonemes = predicted_phonemes.split()
        predicted_sequence_without_sil = [[]]

        # Ensure that we remove the sils
        for pred_phoneme, pred_seq in zip(predicted_phonemes, predicted_sequence[0]):
            if pred_phoneme != "sil":
                predicted_sequence_without_sil[0].append(pred_seq)


        print("Converting canonical to appropriate format for getting error")
        phn_list_canonical = canonical_phonemes.strip().split()
        phn_list_canonical_without_sil = list(filter(lambda phn: phn != "sil", phn_list_canonical))
        phn_encoded_list_canonical = [self.label_encoder.encode_sequence(phn_list_canonical_without_sil)]
        canonicals = torch.LongTensor(phn_encoded_list_canonical)
        canonical_lens = torch.ones((1,1))

        print("Getting the error stats")
        error_metrics = ErrorRateStats()
        error_metrics.append(
                        ids=[test_audio_path],
                        predict=predicted_sequence_without_sil,
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
        return predicted_sequence_without_sil, score, self.extract_stats_from_wer_stats_string(stats_string)

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
            "deletions": {"canonical": {}, "predicted": {}},
            "insertions": {"canonical": {}, "predicted": {}},
            "substitutions": {"canonical": {}, "predicted": {}},
            "canonical": canonical,
            "predicted": predicted
        }

        for i, item in enumerate(zip(canonical, operator, predicted)):
            canonical_phn, op, predicted_phn = item
            if op == "I":
                # errors["insertions"]["canonical"].append((i, canonical_phn))
                # errors["insertions"]["predicted"].append((i, predicted_phn))
                errors["insertions"]["canonical"][i] = canonical_phn
                errors["insertions"]["predicted"][i] = predicted_phn
            elif op == "S":
                # errors["substitutions"]["canonical"].append((i, canonical_phn))
                # errors["substitutions"]["predicted"].append((i, predicted_phn))
                errors["substitutions"]["canonical"][i] = canonical_phn
                errors["substitutions"]["predicted"][i] = predicted_phn
            elif op == "D":
                # errors["deletions"]["canonical"].append((i, canonical_phn))
                # errors["deletions"]["predicted"].append((i, predicted_phn))
                errors["deletions"]["canonical"][i] = canonical_phn
                errors["deletions"]["predicted"][i] = predicted_phn

        return errors


# Get only the label encoder
def get_label_encoder(hparams):
    #NOTE: ensure that the /results folder contains only the checkpoint inside it along with the label encoder

    # Load or compute the label encoder
    lab_enc_file = os.path.join(os.path.normpath(hparams["save_folder"]), "label_encoder.txt")

    label_encoder = CTCTextEncoder()

    #this directly gets the path
    label_encoder.load(
        lab_enc_file
    )

    return label_encoder


def get_wav2vec2_asr_sb_object(hparams_file):
    print(f"Loading hparams from yaml file located at {hparams_file}")

    hparams = None
    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    label_encoder = get_label_encoder(hparams)

    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
        # run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    asr_brain.label_encoder = label_encoder
    print("Wave2Vec2 model object is created with label encoder")
    return asr_brain

# Uncomment for testing locally
if __name__ == '__main__':
    wave2vec2_asr_brain = get_wav2vec2_asr_sb_object('./ml/config/wave2vec2/hparams/inference.yaml')
    test_audio_path = "./assets/arctic_a0100.wav"
    canonical_phonemes = "sil y uw m ah s t s l iy p sil hh iy er jh d sil"  # actual sentence is 'You must sleep he urged'
    predicted_phonemes, score, stats = wave2vec2_asr_brain.evaluate_test_audio(test_audio_path, canonical_phonemes)
    print("Done local testing")