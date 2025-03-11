import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner
from char_indexer import BertCharacterIndexer, VanillaCharacterIndexer
textclenaer = TextCleaner()
char_indexer = VanillaCharacterIndexer()
bert_indexer = BertCharacterIndexer()

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

EPOCH = 8

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import unicodedata
import re
from num2words import num2words
import string
PUNCTUATION = ''.join(sorted(set(';:,.!?¡¿—…"«»“”‘’،؛؟٫٬٪﴾﴿ـ' + string.punctuation)))

def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False

def clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):
            continue
        if is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def convert_numbers_to_arabic_words(text):
    """Convert English numerals in Arabic text to Arabic word form."""
    # Find all numbers in the text with word boundaries
    numbers = re.findall(r'\d+', text)
    
    # Sort numbers by length in descending order to avoid partial replacements
    # (e.g., replacing "19" in "1986" before replacing "1986" itself9
    numbers.sort(key=len, reverse=True)
    
    # Replace each number with its Arabic word form
    for num in numbers:
        try:
            # Convert to integer
            n = int(num)
            # Use num2words with Arabic language
            arabic_word = num2words(n, lang='ar')
            # Replace the number with its word form using word boundaries
            text = re.sub(re.escape(num), arabic_word, text)
        except (ValueError, NotImplementedError):
            # Skip if conversion fails
            continue
    
    return text

def filter_non_arabic_words(text):
    """Remove non-Arabic words from text."""
    # Arabic Unicode range (includes Arabic, Persian, Urdu characters)
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u0660-\u0669]+')
    
    # Split text into words
    words = text.split()
    
    # Keep only words that contain Arabic characters
    arabic_words = []
    for word in words:
        # Check if the word ONLY contains Arabic characters
        if arabic_pattern.search(word):
            arabic_words.append(word)
    
    # Join the Arabic words back into text
    return ' '.join(arabic_words)

def separate_words_and_punctuation(text):
    """
    Separate text into a list of words and punctuation using regex for better performance.
    Punctuation marks are treated as separate tokens.
    """
    # Create a regex pattern that matches either a punctuation character or a non-space, non-punctuation sequence
    # We escape each punctuation character and join them into a character class
    punct_pattern = '|'.join(re.escape(p) for p in PUNCTUATION)
    pattern = f'({punct_pattern})|([^\s{re.escape("".join(PUNCTUATION))}]+)'
    
    # Find all matches
    tokens = re.findall(pattern, text)
    
    # Flatten the list of tuples and remove empty strings
    result = [t[0] if t[0] else t[1] for t in tokens]
    
    return result

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(
    language='ar',
    preserve_punctuation=True, 
    with_stress=True
)

def phonemize(text, global_phonemizer):
    """Convert text to phonemes and token IDs.
    
    Args:
        text: Input text to phonemize
        global_phonemizer: Phonemizer instance
        tokenizer: Tokenizer instance (optional if use_tokenizer=False)
        use_tokenizer: Whether to use tokenizer or simple word separation
        
    Returns:
        Dictionary containing phonemes and optionally token_ids
    """
    # Common preprocessing steps
    text = convert_numbers_to_arabic_words(text)
    text = filter_non_arabic_words(text)
    text = remove_accents(text)
    # text = clean_text(text) # TODO: add in a future PR after testing

    # Tokenization step - either using tokenizer or simple word separation
    tokens = separate_words_and_punctuation(text)
    # Process phonemes without tokenizer-specific handling
    phonemes = [global_phonemizer.phonemize([token], strip=True)[0] 
                if token not in PUNCTUATION else token for token in tokens]

    # Return appropriate result based on tokenization method
    return phonemes

# load phonemizer

config = yaml.safe_load(open("Models/LibriTTS/config.yml"))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert
BERT_path = "Utils/PLBERT/arabic"
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config['model_params'])
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

# path = "/root/notebooks/voiceAI/arabic_audio_ai_fadi/external/style_tts2/Models/LibriTTS/epochs_2nd_00020.pth"
path = f"Models/FineTune.FirstRun/epoch_2nd_{EPOCH-1:05d}.pth"

params_whole = torch.load(path, map_location='cpu')
print(f"Loading checkpoint from: {path}")
params = params_whole['net']

for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model[key].load_state_dict(new_state_dict, strict=False)
_ = [model[key].eval() for key in model]

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)

def inference(text, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    phonemized_text = phonemize(text, global_phonemizer)
    ps = ' '.join(phonemized_text)

    # tokens = textclenaer(ps)
    tokens = char_indexer(ps)
    tokens_bert = bert_indexer(tokens)
    tokens.insert(0, 0)
    tokens_bert.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    tokens_bert = torch.LongTensor(tokens_bert).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens_bert, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)


        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en,
                                         s, input_lengths, text_mask)

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


    return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later


# text = ''' StyleTTS 2 is a text to speech model that leverages style diffusion and adversarial training with large speech language models to achieve human level text to speech synthesis. ''' # @param {type:"string"}
# text = '''انا اسمي فادي''' # @param {type:"string"}
text = '''مرحبا، اسمي فادي وأنا أعمل كمطور برمجيات في شركة تقنية متخصصة في الذكاء الاصطناعي.''' # @param {type:"string}

reference_dicts = {}
reference_dicts['validated_0'] = "Mozilla/wavs/train_0.wav"

noise = torch.randn(1,1,256).to(device)
for k, path in reference_dicts.items():
    ref_s = compute_style(path)
    wav = inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1)

    # Save the synthesized audio and reference
    import os
    os.makedirs('output_audio', exist_ok=True)
    import scipy.io.wavfile as wavfile
    wavfile.write(f'output_audio/epoch_{EPOCH}_synthesized.wav', 24000, wav.astype(np.float32))
    print(f"Saved to: output_audio/epoch_{EPOCH}_synthesized.wav")