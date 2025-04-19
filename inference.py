import random
import time
import yaml
import numpy as np
import torch
import torchaudio
import phonemizer
import argparse
from scipy.io.wavfile import write as write_wav
from collections import OrderedDict
from models import load_ASR_models, load_F0_models, build_model
from utils import recursive_munch
from char_indexer import BertCharacterIndexer, VanillaCharacterIndexer
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

# Parse command line arguments
parser = argparse.ArgumentParser(description='StyleTTS2 Inference')
parser.add_argument('--config', type=str, default="config.yml", help='Path to config file')
parser.add_argument('--model', type=str, default="model.pth", help='Path to model file')
parser.add_argument('--text', type=str, help='Arabic text to synthesize', default="الإِتْقَانُ يَحْتَاجُ إِلَى الْعَمَلِ وَالْمُثَابَرَةِ.")
args = parser.parse_args()

# Set seeds for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(0)
np.random.seed(0)

# Set the HF_HOME environment variable to the path of the Hugging Face cache directory

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='ar',preserve_punctuation=True, with_stress=True)
config = yaml.safe_load(open(args.config))

F0_path = "Utils/JDC/bst.t7"
ASR_config =  "Utils/ASR/config.yml"
ASR_path = "Utils/ASR/epoch_00080.pth"

# load pretrained ASR model
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert
plbert = load_plbert(config['PLBERT_repo_id'], config['PLBERT_dirname'])

model_params = recursive_munch(config['model_params'])
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

state = torch.load(args.model, map_location='cpu')

params = state['net']

for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model[key].load_state_dict(new_state_dict, strict=False)

_ = [model[key].eval() for key in model]


sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)

def inference(phonemes, diffusion_steps=5, embedding_scale=1):
    tokens = VanillaCharacterIndexer()(phonemes)
    bert_tokens = BertCharacterIndexer()(phonemes)
    tokens.insert(0, 0); bert_tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    bert_tokens = torch.LongTensor(bert_tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(bert_tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(
            noise = torch.randn((1, 256)).unsqueeze(1).to(device),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            num_steps=diffusion_steps).squeeze(1)


        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr,
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))


    return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed late

# Generate a short diacritized Arabic sentence about mastery requiring work

print("Arabic sentence:")
print(args.text)

# Phonemize the Arabic sentence
phonemes = global_phonemizer.phonemize([args.text])[0]

print("\nPhonemized text:")
print(phonemes)

noise = torch.randn(1,1,256).to(device)
start = time.time()
wav = inference(phonemes, diffusion_steps=5, embedding_scale=1)
rtf = (time.time() - start) / (len(wav) / 24000)
print(f"RTF = {rtf:5f}")

# Save the synthesized audio to a file
synth_audio_path = "synthesized_audio.wav"
write_wav(synth_audio_path, 24000, wav)
print(f"Synthesized audio saved to {synth_audio_path}")
