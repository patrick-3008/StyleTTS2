import random
import time
import yaml
import numpy as np
import torch
import torchaudio
import phonemizer
import argparse
import logging
from scipy.io.wavfile import write as write_wav
from collections import OrderedDict
from models import load_ASR_models, load_F0_models, build_model
from utils import recursive_munch
from char_indexer import BertCharacterIndexer, VanillaCharacterIndexer
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
import warnings
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('StyleTTS2')

warnings.filterwarnings("ignore", message="dropout option adds dropout after all but last recurrent layer")

def parse_arguments():
    """Parse command line arguments for StyleTTS2 inference."""
    parser = argparse.ArgumentParser(description='StyleTTS2 Inference')
    parser.add_argument('--config', type=str, default="config.yml", help='Path to config file')
    parser.add_argument('--model', type=str, default="model.pth", help='Path to model file')
    parser.add_argument('--text', type=str, 
                        default="الإِتْقَانُ يَحْتَاجُ إِلَى الْعَمَلِ وَالْمُثَابَرَةِ.",
                        help='Arabic text to synthesize')
    parser.add_argument('--output', type=str, default="synthesized_audio.wav", 
                        help='Output audio file path')
    parser.add_argument('--diffusion_steps', type=int, default=5,
                        help='Number of diffusion steps')
    parser.add_argument('--embedding_scale', type=float, default=1.0,
                        help='Embedding scale for diffusion')
    return parser.parse_args()

def set_seeds(seed=0):
    """Set seeds for reproducibility."""
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)        # for current GPU
    torch.cuda.manual_seed_all(seed)    # for all GPUs
    
    # CUDA deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variables (some CUDA operations use these)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.debug(f"Random seeds set to {seed} across all libraries")

def length_to_mask(lengths):
    """Convert lengths tensor to mask tensor."""
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave, mean=-4, std=4):
    """Preprocess audio waveform to mel spectrogram."""
    to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def load_models(config, device):
    """Load all required models for inference."""
    logger.info("Loading models...")
    
    # Load pretrained models
    logger.info("Loading ASR model from Utils/ASR/epoch_00080.pth")
    text_aligner = load_ASR_models("Utils/ASR/epoch_00080.pth", "Utils/ASR/config.yml")
    
    logger.info("Loading F0 model from Utils/JDC/bst.t7")
    pitch_extractor = load_F0_models("Utils/JDC/bst.t7")
    
    # Load BERT model
    logger.info(f"Loading PLBERT model from {config['PLBERT_repo_id']}/{config['PLBERT_dirname']}")
    from Utils.PLBERT.util import load_plbert
    plbert = load_plbert(config['PLBERT_repo_id'], config['PLBERT_dirname'])
    
    # Build the main model
    logger.info("Building StyleTTS2 model")
    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    
    # Move models to device and set to eval mode
    logger.info(f"Moving models to device: {device}")
    for key in model:
        model[key].eval()
        model[key].to(device)
    
    return model, model_params

def load_model_weights(model, model_path, device):
    """Load model weights from checkpoint file."""
    logger.info(f"Loading model weights from {model_path}")
    state = torch.load(model_path, map_location=device)
    params = state['net']
    
    for key in model:
        if key in params:
            logger.info(f"Loading weights for component: {key}")
            try:
                model[key].load_state_dict(params[key])
            except:
                # Handle models saved with DataParallel
                logger.info(f"Handling DataParallel format for {key}")
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    # Remove 'module.' prefix if present, regardless of prefix length
                    name = k.replace('module.', '') if k.startswith('module.') else k
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)
    
    # Ensure all models are in eval mode after loading weights
    for key in model:
        model[key].eval()
    
    logger.info("Model weights loaded successfully")
    return model

def create_diffusion_sampler(model):
    """Create a diffusion sampler for style generation."""
    logger.info("Creating diffusion sampler")
    return DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )

def inference(model, model_params, phonemes, sampler, device, diffusion_steps=5, embedding_scale=1):
    """Generate speech from phonemized text."""
    logger.info("Starting inference process")
    logger.info(f"Parameters: diffusion_steps={diffusion_steps}, embedding_scale={embedding_scale}")
    
    # Tokenize input phonemes
    logger.debug("Tokenizing input phonemes")
    tokens = VanillaCharacterIndexer()(phonemes)
    bert_tokens = BertCharacterIndexer()(phonemes)
    tokens.insert(0, 0)
    bert_tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    bert_tokens = torch.LongTensor(bert_tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        # Prepare input lengths and mask
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)
        
        # Text and BERT encoding
        logger.debug("Performing text and BERT encoding")
        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(bert_tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        
        # Style generation through diffusion
        logger.debug(f"Generating style with {diffusion_steps} diffusion steps")
        s_pred = sampler(
            noise=torch.randn((1, 256)).unsqueeze(1).to(device),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            num_steps=diffusion_steps
        ).squeeze(1)
        
        # Split style vector into style and reference components
        style_vector = s_pred[:, 128:]
        reference_vector = s_pred[:, :128]
        
        # Duration prediction
        logger.debug("Predicting phoneme durations")
        duration_encoding = model.predictor.text_encoder(d_en, style_vector, input_lengths, text_mask)
        lstm_output, _ = model.predictor.lstm(duration_encoding)
        duration_logits = model.predictor.duration_proj(lstm_output)
        
        # Process durations
        duration_probs = torch.sigmoid(duration_logits).sum(axis=-1)
        predicted_durations = torch.round(duration_probs.squeeze()).clamp(min=1)
        
        # Create alignment target
        logger.debug("Creating alignment target")
        alignment_target = torch.zeros(input_lengths, int(predicted_durations.sum().data))
        current_frame = 0
        for i in range(alignment_target.size(0)):
            dur_i = int(predicted_durations[i].data)
            alignment_target[i, current_frame:current_frame + dur_i] = 1
            current_frame += dur_i
        
        # Encode prosody
        logger.debug("Encoding prosody")
        prosody_encoding = (duration_encoding.transpose(-1, -2) @ alignment_target.unsqueeze(0).to(device))
        
        # Handle HifiGAN decoder specifics
        if model_params.decoder.type == "hifigan":
            logger.debug("Applying HifiGAN-specific processing")
            shifted_encoding = torch.zeros_like(prosody_encoding)
            shifted_encoding[:, :, 0] = prosody_encoding[:, :, 0]
            shifted_encoding[:, :, 1:] = prosody_encoding[:, :, 0:-1]
            prosody_encoding = shifted_encoding
        
        # Predict F0 and noise
        logger.debug("Predicting F0 and noise")
        f0_prediction, noise_prediction = model.predictor.F0Ntrain(prosody_encoding, style_vector)
        
        # Prepare ASR features
        logger.debug("Preparing ASR features")
        asr_features = (t_en @ alignment_target.unsqueeze(0).to(device))
        
        # Handle HifiGAN decoder specifics for ASR features
        if model_params.decoder.type == "hifigan":
            shifted_asr = torch.zeros_like(asr_features)
            shifted_asr[:, :, 0] = asr_features[:, :, 0]
            shifted_asr[:, :, 1:] = asr_features[:, :, 0:-1]
            asr_features = shifted_asr
        
        # Generate audio
        logger.debug("Generating audio waveform")
        audio_output = model.decoder(
            asr_features,
            f0_prediction, 
            noise_prediction, 
            reference_vector.squeeze().unsqueeze(0)
        )
    
    logger.info("Inference completed successfully")
    # Remove artifacts at the end of the audio
    return audio_output.squeeze().cpu().numpy()[..., :-50]

def main():
    """Main function for StyleTTS2 inference."""
    # Parse arguments and set up environment
    args = parse_arguments()
    logger.info("Starting StyleTTS2 inference")
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Model file: {args.model}")
    logger.info(f"Output file: {args.output}")
    
    set_seeds()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully")
    
    # Load phonemizer for Arabic
    logger.info("Initializing Arabic phonemizer")
    phonemizer_backend = phonemizer.backend.EspeakBackend(
        language='ar', 
        preserve_punctuation=True, 
        with_stress=True
    )
    
    # Load models
    model, model_params = load_models(config, device)
    model = load_model_weights(model, args.model, device)
    sampler = create_diffusion_sampler(model)
    
    # Process input text
    logger.info("Processing input text")
    logger.info("Arabic sentence: %s", args.text)
    
    # Phonemize the Arabic sentence
    phonemes = phonemizer_backend.phonemize([args.text])[0]
    logger.info("Phonemized text: %s", phonemes)
    
    # Generate speech
    logger.info("Generating speech...")
    start = time.time()
    wav = inference(
        model, 
        model_params, 
        phonemes, 
        sampler, 
        device, 
        diffusion_steps=args.diffusion_steps, 
        embedding_scale=args.embedding_scale
    )
    
    # Calculate real-time factor
    generation_time = time.time() - start
    audio_duration = len(wav) / 24000  # assuming 24kHz sample rate
    rtf = generation_time / audio_duration
    logger.info(f"Generation completed in {generation_time:.2f} seconds")
    logger.info(f"Audio duration: {audio_duration:.2f} seconds")
    logger.info(f"Real-time factor (RTF): {rtf:.5f}")
    
    # Save the synthesized audio to a file
    logger.info(f"Saving audio to {args.output}")
    write_wav(args.output, 24000, wav)
    logger.info("Synthesis completed successfully")

if __name__ == "__main__":
    main()
