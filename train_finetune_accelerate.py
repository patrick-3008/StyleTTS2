# Standard library imports
import os
import random
import time
import warnings

# Third-party imports
import click
import numpy as np
import shutil
import torch
import torch.nn.functional as F
import wandb
import yaml
from accelerate import Accelerator
from munch import Munch

# Suppress warnings for cleaner output
warnings.simplefilter('ignore')

# Local imports
from meldataset import build_dataloader
from Utils.PLBERT.util import load_plbert
from models import *
from losses import *
from utils import *
from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import (
    DiffusionSampler,
    ADPM2Sampler,
    KarrasSchedule
)
from optimizers import build_optimizer

accelerator = Accelerator()

# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
def setup_logging(config_path, log_dir):
    """Set up logging directory and copy config file."""

    config = yaml.safe_load(open(config_path))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(log_dir, os.path.basename(config_path)))
    
    # Initialize wandb
    wandb.init(project="style_tts2_finetune", config=config, dir=log_dir)

def load_pretrained_models(config, device):
    """Load pretrained models (ASR, F0, PLBERT)."""
    # load pretrained ASR model
    text_aligner = load_ASR_models(config['ASR_path'], config['ASR_config'])
    
    # load pretrained F0 model
    pitch_extractor = load_F0_models(config['F0_path'])
    
    # load PL-BERT model
    plbert = load_plbert(config['PLBERT_dir'])
    
    return text_aligner, pitch_extractor, plbert

def build_and_setup_model(model_params, text_aligner, pitch_extractor, plbert, device):
    """Build model and move to device."""
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]
    
    # Apply DataParallel to appropriate model components
    model = Munch({key: MyDataParallel(model[key]) if key not in ["mpd", "msd", "wd"] else model[key] for key in model})
    return model

def setup_dataloaders(data_params, batch_size, device):
    """Set up training and validation dataloaders."""
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    
    train_list, val_list = get_data_path_list(train_path, val_path)
    
    train_dataloader = build_dataloader(
        train_list, 
        batch_size=batch_size, 
        num_workers=2, 
        device=device, 
        **data_params
    )
    
    val_dataloader = build_dataloader(
        val_list, 
        batch_size=batch_size, 
        validation=True, 
        num_workers=0, 
        device=device, 
        **data_params
    )
    
    return train_dataloader, val_dataloader

def setup_optimizer(model, optimizer_params, epochs, train_dataloader):
    """Set up and configure the optimizer with appropriate learning rates."""
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['bert']['max_lr'] = optimizer_params.bert_lr * 2
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2
    
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                               scheduler_params_dict=scheduler_params_dict, lr=optimizer_params.lr)
    
    # adjust BERT learning rate
    for g in optimizer.optimizers['bert'].param_groups:
        g['betas'] = (0.9, 0.99)
        g['lr'] = optimizer_params.bert_lr
        g['initial_lr'] = optimizer_params.bert_lr
        g['min_lr'] = 0
        g['weight_decay'] = 0.01
        
    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99)
            g['lr'] = optimizer_params.ft_lr
            g['initial_lr'] = optimizer_params.ft_lr
            g['min_lr'] = 0
            g['weight_decay'] = 1e-4
            
    return optimizer

def setup_losses_and_sampler(model, model_params, slmadv_params, sr, device, sampler):
    """Set up loss functions and diffusion sampler."""
    generator_loss = MyDataParallel(GeneratorLoss(model.mpd, model.msd).to(device))
    discriminator_loss = MyDataParallel(DiscriminatorLoss(model.mpd, model.msd).to(device))
    wavlm_loss = MyDataParallel(WavLMLoss(model_params.slm.model, model.wd, sr, model_params.slm.sr).to(device))
    stft_loss = MultiResolutionSTFTLoss().to(device)
    
   
    slmadv = SLMAdversarialLoss(model, wavlm_loss, sampler, **slmadv_params)
    
    return generator_loss, discriminator_loss, wavlm_loss, stft_loss, slmadv, sampler

def compute_reference_styles(model, ref_mels):
    """Compute reference styles for multispeaker training with no gradient tracking."""
    with torch.no_grad():
        ref_acoustic_style = model.style_encoder(ref_mels.unsqueeze(1))
        ref_prosodic_style = model.predictor_encoder(ref_mels.unsqueeze(1))
        return torch.cat([ref_acoustic_style, ref_prosodic_style], dim=1)

def create_masks(mel_input_length, input_lengths, n_down, device):
    """Create masks for text and mel inputs based on their lengths."""
    with torch.no_grad():
        mel_mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
        text_mask = length_to_mask(input_lengths).to(device)
    return mel_mask, text_mask

def process_attention_matrix(s2s_attn):
    """
    Process the attention matrix from the text aligner.
    
    This function:
    1. Transposes the attention matrix to swap the last two dimensions
    2. Removes the first frame/token (likely a start token or padding)
    3. Transposes back to the original dimension order
    
    Args:
        s2s_attn: The speech-to-text attention matrix from the text aligner
        
    Returns:
        Processed attention matrix with the first token removed
    """
    # Swap the last two dimensions (batch_size, heads, text_len, mel_len) -> (batch_size, heads, mel_len, text_len)
    s2s_attn = s2s_attn.transpose(-1, -2)
    
    # Remove the first token/frame (likely a start token or padding)
    # This slices out everything except the first element in the last dimension
    s2s_attn = s2s_attn[..., 1:]
    
    # Transpose back to original dimension order
    # (batch_size, heads, mel_len, text_len) -> (batch_size, heads, text_len, mel_len)
    s2s_attn = s2s_attn.transpose(-1, -2)
    
    return s2s_attn

def train_diffusion(model, model_params, sampler, s_trg, bert_dur, ref_mels=None, device=None):
    """
    Train the diffusion model component.
    
    Args:
        model: The model containing the diffusion component
        model_params: Model parameters
        sampler: The diffusion sampler
        s_trg: Target style vector
        bert_dur: BERT duration embeddings
        ref_mels: Reference mel spectrograms (for multispeaker training)
        device: The device to run on
        
    Returns:
        tuple: (loss_diff, loss_sty, sigma_data) - diffusion loss, style loss, and sigma data if estimated
    """
    num_steps = np.random.randint(3, 5)
    sigma_data = None
    
    if model_params.diffusion.dist.estimate_sigma_data:
        sigma_data = s_trg.std(axis=-1).mean().item()  # batch-wise std estimation
        model.diffusion.module.diffusion.sigma_data = sigma_data
    
    multispeaker = model_params.multispeaker
    if multispeaker:
        reference_styles = compute_reference_styles(model, ref_mels)
        
        s_preds = sampler(
            noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
            embedding=bert_dur,
            embedding_scale=1,
            features=reference_styles,  # reference from the same speaker as the embedding
            embedding_mask_proba=0.1,
            num_steps=num_steps
        ).squeeze(1)
        
        loss_diff = model.diffusion(
            s_trg.unsqueeze(1), 
            embedding=bert_dur, 
            features=reference_styles
        ).mean()  # EDM loss
    else:
        s_preds = sampler(
            noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
            embedding=bert_dur,
            embedding_scale=1,
            embedding_mask_proba=0.1,
            num_steps=num_steps
        ).squeeze(1)
        
        loss_diff = model.diffusion.module.diffusion(
            s_trg.unsqueeze(1), 
            embedding=bert_dur
        ).mean()  # EDM loss
    
    loss_sty = F.l1_loss(s_preds, s_trg.detach())  # style reconstruction loss
    
    return loss_diff, loss_sty, sigma_data

def align_text_to_speech(model, texts, input_lengths, mel_input_length, s2s_attn, text_mask, n_down):
    """
    Aligns encoded text to speech using attention matrices and computes durations.
    
    Args:
        model: The model containing text_encoder
        texts: Input text tokens
        input_lengths: Lengths of input texts
        mel_input_length: Lengths of mel spectrograms
        s2s_attn: Speech-to-text attention matrix
        text_mask: Mask for text inputs
        n_down: Downsampling factor
        
    Returns:
        tuple: (aligned_text_features, monotonic_attention, duration_gt) - 
               aligned text features, monotonic attention matrix, and ground truth durations
    """
    mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
    monotonic_attention = maximum_path(s2s_attn, mask_ST)

    # encode text
    encoded_text = model.text_encoder(texts, input_lengths, text_mask)
    
    # 50% chance of using monotonic version of attention
    if bool(random.getrandbits(1)):
        aligned_text_features = (encoded_text @ s2s_attn)
    else:
        aligned_text_features = (encoded_text @ monotonic_attention)

    duration_gt = monotonic_attention.sum(axis=-1).detach()
    
    return aligned_text_features, monotonic_attention, duration_gt

def compute_global_styles(model, mels, mel_input_length):
    """
    Compute global acoustic and prosodic styles for each mel spectrogram in the batch.
    
    This operation cannot be done in batch because of the avgpool layer.
    
    Args:
        model: The model containing style_encoder and predictor_encoder
        mels: Batch of mel spectrograms
        mel_input_length: Lengths of mel spectrograms
        
    Returns:
        tuple: (s_dur, gs, s_trg) - prosodic styles, acoustic styles, and combined target styles
    """
    prosodic_styles = []
    acoustic_styles = []
    
    for i in range(len(mel_input_length)):
        mel = mels[i, :, :mel_input_length[i]]
        
        # Extract prosodic style
        prosodic_style = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
        prosodic_styles.append(prosodic_style)
        
        # Extract acoustic style
        acoustic_style = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
        acoustic_styles.append(acoustic_style)

    # Stack and squeeze properly to maintain batch dimension
    prosodic_styles = torch.stack(prosodic_styles)
    prosodic_styles = prosodic_styles.squeeze(tuple(range(1, len(prosodic_styles.shape))))  # global prosodic styles
    
    acoustic_styles = torch.stack(acoustic_styles)
    acoustic_styles = acoustic_styles.squeeze(tuple(range(1, len(acoustic_styles.shape))))  # global acoustic styles
    
    # Combine styles for denoiser ground truth
    target_styles = torch.cat([acoustic_styles, prosodic_styles], dim=-1).detach()
    
    return prosodic_styles, target_styles

def encode_text_with_bert(model, texts, text_mask):
    """
    Encode text inputs using BERT and the BERT encoder.
    
    Args:
        model: The model containing bert and bert_encoder components
        texts: Input text tokens
        text_mask: Mask for text inputs
        
    Returns:
        Encoded BERT output with transposed dimensions for further processing
    """
    bert_output = model.bert(texts, attention_mask=(~text_mask).int())
    bert_output_encoded = model.bert_encoder(bert_output).transpose(-1, -2)
    return bert_output, bert_output_encoded

def extract_training_segments(mel_input_length, aligned_text, p, mels, waves, device, max_len):
    """
    Extract random segments from mel spectrograms and audio waveforms for training.
    
    Args:
        mel_input_length: Lengths of mel spectrograms in the batch
        aligned_text: Text features aligned with speech
        p: Predicted features from the predictor
        mels: Mel spectrograms
        waves: Audio waveforms
        device: Device to place tensors on
        max_len: Maximum length constraint for segments
        
    Returns:
        tuple: (
            aligned_text_segments,
            predicted_feature_segments,
            ground_truth_mel_segments,
            waveform_segments,
        )
    """
    # Calculate segment lengths
    content_segment_length = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
    
    # Initialize empty lists for collecting segments
    aligned_text_segments = []
    ground_truth_mel_segments = []
    predicted_feature_segments = []
    ground_truth_waveform_segments = []
    
    # Extract segments for each sample in the batch
    for batch_idx in range(len(mel_input_length)):
        mel_length = int(mel_input_length[batch_idx].item() / 2)

        # Select random starting point for content segments
        content_random_start = np.random.randint(0, mel_length - content_segment_length)
        
        # Extract content segments
        aligned_text_segments.append(aligned_text[batch_idx, :, content_random_start:content_random_start+content_segment_length])
        predicted_feature_segments.append(p[batch_idx, :, content_random_start:content_random_start+content_segment_length])
        ground_truth_mel_segments.append(mels[batch_idx, :, (content_random_start * 2):((content_random_start+content_segment_length) * 2)])
        
        # Extract corresponding audio segment (300 is the ratio of audio samples to mel frames)
        audio_segment = waves[batch_idx][(content_random_start * 2) * 300:((content_random_start+content_segment_length) * 2) * 300]
        ground_truth_waveform_segments.append(torch.from_numpy(audio_segment).to(device))
        
    # Stack segments into tensors
    ground_truth_waveform_segments = torch.stack(ground_truth_waveform_segments).float().detach()
    aligned_text_segments = torch.stack(aligned_text_segments)
    predicted_feature_segments = torch.stack(predicted_feature_segments)
    ground_truth_mel_segments = torch.stack(ground_truth_mel_segments).detach()
    
    return (
        aligned_text_segments,
        predicted_feature_segments,
        ground_truth_mel_segments,
        ground_truth_waveform_segments
        )

@click.command()
@click.option('-p', '--config_path', default='Configs/config_ft.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))
    
    # Extract configuration parameters
    log_dir = config['log_dir']
    batch_size = config['batch_size']
    epochs = config['epochs']
    save_freq = config['save_freq']
    log_interval = config['log_interval']
    data_params = config['data_params']
    sr = config['preprocess_params']['sr']
    max_len = config['max_len']
    
    loss_params = Munch(config['loss_params'])
    diffusion_training_epoch = loss_params.diffusion_training_epoch
    joint_training_epoch = loss_params.joint_training_epoch
    
    optimizer_params = Munch(config['optimizer_params'])
    
    # Set up logging
    setup_logging(config_path, log_dir)
    
    # Get device from accelerator
    device = accelerator.device
    
    # Set up dataloaders
    train_dataloader, val_dataloader = setup_dataloaders(data_params, batch_size, device)
    
    # Load pretrained models
    text_aligner, pitch_extractor, plbert = load_pretrained_models(config, device)
    
    # Build and setup model
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model = build_and_setup_model(model_params, text_aligner, pitch_extractor, plbert, device)

    # Create sampler first so it can be used by SLMAdversarialLoss
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )
 
    # Set up losses and sampler
    slmadv_params = config['slmadv_params']
    generator_loss, discriminator_loss, wavlm_loss, stft_loss, slmadv, sampler = setup_losses_and_sampler(
        model, model_params, slmadv_params, sr, device, sampler
    )
    
    # Set up optimizer with appropriate learning rates
    optimizer = setup_optimizer(model, optimizer_params, epochs, train_dataloader)
    model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, config['pretrained_model'], load_only_params=config['load_only_params'])

    n_down = model.text_aligner.n_down

    best_loss = float('inf'); iters = 0; running_std = []
   
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    torch.cuda.empty_cache()

    for epoch in range(start_epoch, epochs):
        running_loss = 0

        _ = [model[key].eval() for key in model]
        _ = [model[key].train() for key in ['text_aligner', 'text_encoder', 'predictor', 'bert_encoder', 'bert', 'msd', 'mpd']]
        
        for i, batch in enumerate(train_dataloader):
            _ = [b.to(device) for i, b in enumerate(batch) if i > 0]
            waves, texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

            mel_mask, text_mask = create_masks(mel_input_length, input_lengths, n_down, device)

            try:
                _, s2s_pred, s2s_attn = model.text_aligner(mels, mel_mask, texts)
                s2s_attn = process_attention_matrix(s2s_attn)
            except:
                continue

            aligned_text, s2s_attn_mono, d_gt = align_text_to_speech(model, texts, input_lengths, mel_input_length, s2s_attn, text_mask, n_down)

            # Compute global styles
            prosodic_styles, target_styles = compute_global_styles(model, mels, mel_input_length)

            bert_output, bert_output_encoded = encode_text_with_bert(model, texts, text_mask)
            
            # denoiser training
            if epoch >= diffusion_training_epoch:
                loss_diff, loss_sty, sigma_data = train_diffusion(model, model_params, sampler, target_styles, bert_output, ref_mels if multispeaker else None, device)
                
                if sigma_data is not None:
                    running_std.append(sigma_data)
            else:
                loss_sty = 0
                loss_diff = 0

            d, p = model.predictor(bert_output_encoded, prosodic_styles, input_lengths, s2s_attn_mono, text_mask)
                
            aligned_text_segments, predicted_feature_segments, ground_truth_mel_segments, ground_truth_waveform_segments = extract_training_segments(mel_input_length, aligned_text, p, mels, waves, device, max_len)

            if ground_truth_mel_segments.size(-1) < 80:
                continue
            
            acoustic_segment_style = model.style_encoder(ground_truth_mel_segments.unsqueeze(1))           
            prosodic_segment_style = model.predictor_encoder(ground_truth_mel_segments.unsqueeze(1))
                
            with torch.no_grad():
                F0_real, _, F0 = model.pitch_extractor(ground_truth_mel_segments.unsqueeze(1))
                F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze()

                N_real = log_norm(ground_truth_mel_segments.unsqueeze(1)).squeeze(1)
                
                ground_truth_waveform_segments = ground_truth_waveform_segments.unsqueeze(1)
                reconstructed_waveform_segments = model.decoder(aligned_text_segments, F0_real, N_real, acoustic_segment_style)

            F0_fake, N_fake = model.predictor.F0Ntrain(predicted_feature_segments, prosodic_segment_style)

            reconstructed_waveform_all_the_way = model.decoder(aligned_text_segments, F0_fake, N_fake, acoustic_segment_style)

            loss_F0_rec =  (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

            optimizer.zero_grad()
            d_loss = discriminator_loss(ground_truth_waveform_segments, reconstructed_waveform_segments.detach()).mean()
            accelerator.backward(d_loss)
            optimizer.step('msd')
            optimizer.step('mpd')

            # generator loss
            optimizer.zero_grad()

            loss_mel = stft_loss(reconstructed_waveform_all_the_way, ground_truth_waveform_segments)
            loss_gen_all = generator_loss(ground_truth_waveform_segments, reconstructed_waveform_all_the_way).mean()
            loss_lm = wavlm_loss(
                ground_truth_waveform_segments.detach().squeeze(tuple(range(1, len(ground_truth_waveform_segments.shape)))), 
                reconstructed_waveform_all_the_way.squeeze(tuple(range(1, len(reconstructed_waveform_all_the_way.shape))))
            ).mean()

            loss_ce = 0
            loss_dur = 0
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p in range(_s2s_trg.shape[0]):
                    _s2s_trg[p, :_text_input[p]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], 
                                       _text_input[1:_text_length-1])
                loss_ce += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())

            loss_ce /= texts.size(0)
            loss_dur /= texts.size(0)
            
            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= texts.size(0)

            loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10

            g_loss = loss_params.lambda_mel * loss_mel + \
                     loss_params.lambda_F0 * loss_F0_rec + \
                     loss_params.lambda_ce * loss_ce + \
                     loss_params.lambda_norm * loss_norm_rec + \
                     loss_params.lambda_dur * loss_dur + \
                     loss_params.lambda_gen * loss_gen_all + \
                     loss_params.lambda_slm * loss_lm + \
                     loss_params.lambda_sty * loss_sty + \
                     loss_params.lambda_diff * loss_diff + \
                    loss_params.lambda_mono * loss_mono + \
                    loss_params.lambda_s2s * loss_s2s
            
            running_loss += loss_mel.item()
            accelerator.backward(g_loss)

            optimizer.step('bert_encoder')
            optimizer.step('bert')
            optimizer.step('predictor')
            optimizer.step('predictor_encoder')
            optimizer.step('style_encoder')
            optimizer.step('decoder')
            
            optimizer.step('text_encoder')
            optimizer.step('text_aligner')
            
            if epoch >= diffusion_training_epoch:
                optimizer.step('diffusion')

            d_loss_slm, loss_gen_lm = 0, 0
            if epoch >= joint_training_epoch:
                # randomly pick whether to use in-distribution text
                if np.random.rand() < 0.5:
                    use_ind = True
                else:
                    use_ind = False

                if use_ind:
                    ref_lengths = input_lengths
                    ref_texts = texts
                    
                slm_out = slmadv(i, 
                                 ground_truth_waveform_segments, 
                                 reconstructed_waveform_segments, 
                                 waves, 
                                 mel_input_length,
                                 ref_texts, 
                                 ref_lengths, use_ind, target_styles, compute_reference_styles(model, ref_mels) if multispeaker else None)

                if slm_out is not None:
                    d_loss_slm, loss_gen_lm, _ = slm_out

                    # SLM generator loss
                    optimizer.zero_grad()
                    accelerator.backward(loss_gen_lm)

                    # compute the gradient norm
                    total_norm = {}
                    for key in model.keys():
                        total_norm[key] = 0
                        parameters = [p for p in model[key].parameters() if p.grad is not None and p.requires_grad]
                        for p in parameters:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm[key] += param_norm.item() ** 2
                        total_norm[key] = total_norm[key] ** 0.5

                    # gradient scaling
                    if total_norm['predictor'] > slmadv_params['thresh']:
                        for key in model.keys():
                            for p in model[key].parameters():
                                if p.grad is not None:
                                    p.grad *= (1 / total_norm['predictor'])

                    for p in model.predictor.duration_proj.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params['scale']

                    for p in model.predictor.lstm.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params['scale']

                    for p in model.diffusion.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params['scale']
                    
                    optimizer.step('bert_encoder')
                    optimizer.step('bert')
                    optimizer.step('predictor')
                    optimizer.step('diffusion')

                    # SLM discriminator loss
                    if d_loss_slm != 0:
                        optimizer.zero_grad()
                        accelerator.backward(d_loss_slm)
                        optimizer.step('wd')

            iters = iters + 1
            
            if (i+1)%log_interval == 0:
                # Log metrics to wandb
                wandb.log({
                    'train/mel_loss': running_loss / log_interval,
                    'train/gen_loss': loss_gen_all,
                    'train/d_loss': d_loss,
                    'train/ce_loss': loss_ce,
                    'train/dur_loss': loss_dur,
                    'train/slm_loss': loss_lm,
                    'train/norm_loss': loss_norm_rec,
                    'train/F0_loss': loss_F0_rec,
                    'train/sty_loss': loss_sty,
                    'train/diff_loss': loss_diff,
                    'train/d_loss_slm': d_loss_slm,
                    'train/gen_loss_slm': loss_gen_lm,
                })
                
                running_loss = 0
            
        loss_test = 0
        loss_align = 0
        loss_f = 0
        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            iters_test = 0
            for _, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                try:
                    waves = batch[0]
                    batch = [b.to(device) for b in batch[1:]]
                    texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch
                    
                    mel_mask, text_mask = create_masks(mel_input_length, input_lengths, n_down, device)

                    _, _, s2s_attn = model.text_aligner(mels, mel_mask, texts)
                    s2s_attn = process_attention_matrix(s2s_attn)

                    aligned_text, s2s_attn_mono, d_gt = align_text_to_speech(model, texts, input_lengths, mel_input_length, s2s_attn, text_mask, n_down)

                    # Compute global styles
                    prosodic_styles, target_styles = compute_global_styles(model, mels, mel_input_length)

                    _, bert_output_encoded = encode_text_with_bert(model, texts, text_mask)
                    d, p = model.predictor(bert_output_encoded, s, input_lengths, s2s_attn_mono, text_mask)
                    # get clips
                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en = []
                    gt = []

                    p_en = []
                    wav = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)

                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(aligned_text[bib, :, random_start:random_start+mel_len])
                        p_en.append(p[bib, :, random_start:random_start+mel_len])

                        gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                        y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                        wav.append(torch.from_numpy(y).to(device))

                    wav = torch.stack(wav).float().detach()

                    en = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()
                    s = model.predictor_encoder(gt.unsqueeze(1))

                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s)

                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for bib in range(_s2s_trg.shape[0]):
                            _s2s_trg[bib, :_text_input[bib]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], 
                                               _text_input[1:_text_length-1])

                    loss_dur /= texts.size(0)

                    s = model.style_encoder(gt.unsqueeze(1))

                    y_rec = model.decoder(en, F0_fake, N_fake, s)
                    loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1)) 

                    loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                    loss_test += (loss_mel).mean()
                    loss_align += (loss_dur).mean()
                    loss_f += (loss_F0).mean()

                    iters_test += 1
                except:
                    continue

        # Log metrics to wandb instead of tensorboard
        wandb.log({
            'eval/mel_loss': loss_test / iters_test,
            'eval/dur_loss': loss_align / iters_test,
            'eval/F0_loss': loss_f / iters_test
        }, step=epoch + 1)
        
        if (epoch + 1) % save_freq == 0 :
            if (loss_test / iters_test) < best_loss:
                best_loss = loss_test / iters_test
            print('Saving..')
            state = {
                'net':  {key: model[key].state_dict() for key in model}, 
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': loss_test / iters_test,
                'epoch': epoch,
            }
            save_path = os.path.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
            torch.save(state, save_path)

            # if estimate sigma, save the estimated simga
            if model_params.diffusion.dist.estimate_sigma_data:
                config['model_params']['diffusion']['dist']['sigma_data'] = float(np.mean(running_std))

                with open(os.path.join(log_dir, os.path.basename(config_path)), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)

                            
if __name__=="__main__":
    main()
