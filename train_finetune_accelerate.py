# load packages
import random
import yaml
from munch import Munch
import numpy as np
import torch
import torch.nn.functional as F
import click
import shutil
import warnings
warnings.simplefilter('ignore')
import wandb

from meldataset import build_dataloader

from Utils.PLBERT.util import load_plbert

from models import *
from losses import *
from utils import *

from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from optimizers import build_optimizer

from accelerate import Accelerator

accelerator = Accelerator()

# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def load_pretrained_models(config, device):
    """
    Load all pretrained models required for training.
    
    Args:
        config: Configuration dictionary containing model paths
        device: Device to load the models to
        
    Returns:
        tuple: (text_aligner, pitch_extractor, plbert)
    """
    # load pretrained ASR model
    ASR_config = config['ASR_config']
    ASR_path = config['ASR_path']
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    
    # load pretrained F0 model
    F0_path = config['F0_path']
    pitch_extractor = load_F0_models(F0_path)
    
    # load PL-BERT model
    BERT_path = config['PLBERT_dir']
    plbert = load_plbert(BERT_path)
    
    return text_aligner, pitch_extractor, plbert

def setup_optimizers(model, optimizer_params, epochs, train_dataloader_length):
    """
    Set up optimizers with appropriate learning rates and parameters for different model components.
    
    Args:
        model: Dictionary containing model components
        optimizer_params: Parameters for optimizer configuration
        epochs: Total number of training epochs
        train_dataloader_length: Length of the training dataloader
        
    Returns:
        optimizer: Configured optimizer object
    """
    # Base scheduler parameters
    base_scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": train_dataloader_length,
    }

    # Create scheduler params dictionary for each model component
    scheduler_params_dict = {key: base_scheduler_params.copy() for key in model}
    
    # Set specific learning rates for certain components
    scheduler_params_dict['bert']['max_lr'] = optimizer_params.bert_lr * 2
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2
    
    # Build optimizer with model parameters and scheduler parameters
    optimizer = build_optimizer(
        {key: model[key].parameters() for key in model},
        scheduler_params_dict=scheduler_params_dict, 
        lr=optimizer_params.lr
    )
    
    # Configure BERT-specific optimizer parameters
    for param_group in optimizer.optimizers['bert'].param_groups:
        param_group['betas'] = (0.9, 0.99)
        param_group['lr'] = optimizer_params.bert_lr
        param_group['initial_lr'] = optimizer_params.bert_lr
        param_group['min_lr'] = 0
        param_group['weight_decay'] = 0.01
        
    # Configure acoustic module optimizer parameters
    acoustic_modules = ["decoder", "style_encoder"]
    for module_name in acoustic_modules:
        for param_group in optimizer.optimizers[module_name].param_groups:
            param_group['betas'] = (0.0, 0.99)
            param_group['lr'] = optimizer_params.ft_lr
            param_group['initial_lr'] = optimizer_params.ft_lr
            param_group['min_lr'] = 0
            param_group['weight_decay'] = 1e-4
    
    return optimizer

def perform_text_alignment(model, mels, mask, texts):
    """
    Perform text alignment using the text aligner model.
    
    Args:
        model: Model containing the text_aligner component
        mels: Mel spectrograms
        mask: Mask for the mel spectrograms
        texts: Text inputs
        
    Returns:
        tuple: (s2s_pred, s2s_attn) - speech-to-speech predictions and attention
        
    Raises:
        Exception: If text alignment fails
    """
    _, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)
    s2s_attn = s2s_attn.transpose(-1, -2)
    s2s_attn = s2s_attn[..., 1:]
    s2s_attn = s2s_attn.transpose(-1, -2)
    
    return s2s_pred, s2s_attn

def calculate_duration_and_ce_losses(duration_predictions, duration_targets, input_lengths):
    """
    Calculate duration and cross-entropy losses for text-to-speech alignment.
    
    Args:
        duration_predictions: Predicted durations from the model
        duration_targets: Ground truth duration targets
        input_lengths: Lengths of the input sequences
        
    Returns:
        tuple: (loss_ce, loss_dur) - Cross-entropy loss and duration loss
    """
    loss_ce = 0
    loss_dur = 0
    batch_size = len(duration_predictions)
    
    for pred, target, length in zip(duration_predictions, duration_targets, input_lengths):
        # Trim predictions to actual sequence length
        pred = pred[:length, :]
        target = target[:length].long()
        
        # Create binary target matrix
        binary_target = torch.zeros_like(pred)
        for i in range(binary_target.shape[0]):
            binary_target[i, :target[i]] = 1
        
        # Calculate duration prediction by applying sigmoid and summing
        dur_pred = torch.sigmoid(pred).sum(axis=1)
        
        # Skip first and last tokens for duration loss (typically BOS/EOS tokens)
        loss_dur += F.l1_loss(dur_pred[1:length-1], target[1:length-1])
        
        # Calculate cross-entropy loss on flattened predictions and targets
        loss_ce += F.binary_cross_entropy_with_logits(pred.flatten(), binary_target.flatten())
    
    # Normalize losses by batch size
    loss_ce /= batch_size
    loss_dur /= batch_size
    
    return loss_ce, loss_dur

@click.command()
@click.option('-p', '--config_path', default='Configs/config_ft.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))
    
    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    
    # Initialize wandb
    wandb.init(project="test", config=config)

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)
    
    batch_size = config['batch_size']

    epochs = config['epochs']
    save_freq = config['save_freq']
    log_interval = config['log_interval']

    data_params = config['data_params']
    sr = config['preprocess_params']['sr']
    train_path = data_params['train_data']
    val_path = data_params['val_data']

    max_len = config['max_len']
    
    loss_params = Munch(config['loss_params'])
    diffusion_training_epoch = loss_params.diffusion_training_epoch
    joint_training_epoch = loss_params.joint_training_epoch
    
    optimizer_params = Munch(config['optimizer_params'])
    
    train_list, val_list = get_data_path_list(train_path, val_path)
    device = accelerator.device

    train_dataloader = build_dataloader(train_list, batch_size=batch_size, num_workers=2, device=device, **data_params)
    val_dataloader = build_dataloader(val_list, validation=True, batch_size=batch_size, num_workers=0, device=device, **data_params)
    
    # Load pretrained models
    text_aligner, pitch_extractor, plbert = load_pretrained_models(config, device)
    
    # build model
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]
    
    # DP
    for key in model:
        if key != "mpd" and key != "msd" and key != "wd":
            model[key] = MyDataParallel(model[key])
            
    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)
    
    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model, model.wd, sr, model_params.slm.sr).to(device)

    gl = MyDataParallel(gl)
    dl = MyDataParallel(dl)
    wl = MyDataParallel(wl)
    
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )
    
    optimizer = setup_optimizers(
        model, 
        optimizer_params, 
        epochs, 
        len(train_dataloader)
    )
    
    # load models if there is a model
    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, config['pretrained_model'],
                                    load_only_params=config.get('load_only_params', True), ignore_modules=['bert'])

    n_down = model.text_aligner.n_down

    best_loss = float('inf'); iters = 0
    
    torch.cuda.empty_cache()
    
    stft_loss = MultiResolutionSTFTLoss().to(device)
    
    running_std = []
    
    slmadv_params = Munch(config['slmadv_params'])
    slmadv = SLMAdversarialLoss(model, wl, sampler, 
                                slmadv_params.min_len, 
                                slmadv_params.max_len,
                                batch_percentage=slmadv_params.batch_percentage,
                                skip_update=slmadv_params.iter, 
                                sig=slmadv_params.sig
                               )
    
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    for epoch in range(start_epoch, epochs):
        running_loss = 0

        _ = [model[key].eval() for key in model]
        _ = [model[key].train() for key in ['text_aligner', 'text_encoder', 'predictor', 'bert_encoder', 'bert', 'msd', 'mpd']]
        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, bert_texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

            if mels.size(-1) < 80:
                continue

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

                # compute reference styles
                if multispeaker and epoch >= diffusion_training_epoch:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref = torch.cat([ref_ss, ref_sp], dim=1)
                
            try:
                s2s_pred, s2s_attn = perform_text_alignment(model, mels, mask, texts)
            except:
                continue

            mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            # encode
            t_en = model.text_encoder(texts, input_lengths, text_mask)
            
            # 50% of chance of using monotonic version
            if bool(random.getrandbits(1)):
                asr = (t_en @ s2s_attn)
            else:
                asr = (t_en @ s2s_attn_mono)

            d_gt = s2s_attn_mono.sum(axis=-1).detach()

            # compute the style of the entire utterance
            # this operation cannot be done in batch because of the avgpool layer (may need to work on masked avgpool)
            ss = []
            gs = []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item())
                mel = mels[bib, :, :mel_input_length[bib]]
                s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                ss.append(s)
                s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                gs.append(s)

            # Fix: Use squeeze(1) instead of squeeze() to avoid removing batch dimension
            s_dur = torch.stack(ss); s_dur = s_dur.squeeze(tuple(range(1, len(s_dur.shape))))  # global prosodic styles
            gs = torch.stack(gs); gs = gs.squeeze(tuple(range(1, len(gs.shape)))) # global acoustic styles
            s_trg = torch.cat([gs, s_dur], dim=-1).detach() # ground truth for denoiser

            bert_dur = model.bert(bert_texts, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
            
            # denoiser training
            if epoch >= diffusion_training_epoch:
                num_steps = np.random.randint(3, 5)
                
                if model_params.diffusion.dist.estimate_sigma_data:
                    model.diffusion.module.diffusion.sigma_data = s_trg.std(axis=-1).mean().item() # batch-wise std estimation
                    running_std.append(model.diffusion.module.diffusion.sigma_data)
                    
                if multispeaker:
                    s_preds = sampler(noise = torch.randn_like(s_trg).unsqueeze(1).to(device), 
                          embedding=bert_dur,
                          embedding_scale=1,
                                   features=ref, # reference from the same speaker as the embedding
                             embedding_mask_proba=0.1,
                             num_steps=num_steps).squeeze(1)
                    loss_diff = model.diffusion(s_trg.unsqueeze(1), embedding=bert_dur, features=ref).mean() # EDM loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach()) # style reconstruction loss
                else:
                    s_preds = sampler(noise = torch.randn_like(s_trg).unsqueeze(1).to(device), 
                          embedding=bert_dur,
                          embedding_scale=1,
                             embedding_mask_proba=0.1,
                             num_steps=num_steps).squeeze(1)                    
                    loss_diff = model.diffusion.module.diffusion(s_trg.unsqueeze(1), embedding=bert_dur).mean() # EDM loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach()) # style reconstruction loss
            else:
                loss_sty = 0
                loss_diff = 0

                
            d, p = model.predictor(d_en, s_dur, 
                                                    input_lengths, 
                                                    s2s_attn_mono, 
                                                    text_mask)
                
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            import pdb; pdb.set_trace()
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            en = []
            gt = []
            p_en = []
            wav = []
            st = []
            
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start+mel_len])
                p_en.append(p[bib, :, random_start:random_start+mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                
                y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))
                
                # style reference (better to be different from the GT)
                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])
                
            wav = torch.stack(wav).float().detach()

            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()
            
            
            if gt.size(-1) < 80:
                continue
            
            s = model.style_encoder(gt.unsqueeze(1))           
            s_dur = model.predictor_encoder(gt.unsqueeze(1))
                
            with torch.no_grad():
                F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze()

                N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
                
                y_rec_gt = wav.unsqueeze(1)
                y_rec_gt_pred = model.decoder(en, F0_real, N_real, s)

                wav = y_rec_gt

            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)

            y_rec = model.decoder(en, F0_fake, N_fake, s)

            loss_F0_rec =  (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

            optimizer.zero_grad()
            d_loss = dl(wav.detach(), y_rec.detach()).mean()
            accelerator.backward(d_loss)
            optimizer.step('msd')
            optimizer.step('mpd')

            # generator loss
            optimizer.zero_grad()

            loss_mel = stft_loss(y_rec, wav)
            loss_gen_all = gl(wav, y_rec).mean()
            loss_lm = wl(wav.detach().squeeze(tuple(range(1, len(wav.shape)))), y_rec.squeeze(tuple(range(1, len(y_rec.shape))))).mean()

            loss_ce, loss_dur = calculate_duration_and_ce_losses(d, d_gt, input_lengths)
            
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
                                 y_rec_gt, 
                                 y_rec_gt_pred, 
                                 waves, 
                                 mel_input_length,
                                 ref_texts, 
                                 ref_lengths, use_ind, s_trg.detach(), ref if multispeaker else None)

                if slm_out is not None:
                    d_loss_slm, loss_gen_lm, y_pred = slm_out

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
                    if total_norm['predictor'] > slmadv_params.thresh:
                        for key in model.keys():
                            for p in model[key].parameters():
                                if p.grad is not None:
                                    p.grad *= (1 / total_norm['predictor'])

                    for p in model.predictor.duration_proj.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    for p in model.predictor.lstm.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    for p in model.diffusion.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale
                    
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
                logger.info ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Disc Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, LM Loss: %.5f, Gen Loss: %.5f, Sty Loss: %.5f, Diff Loss: %.5f, DiscLM Loss: %.5f, GenLM Loss: %.5f, S2S Loss: %.5f, Mono Loss: %.5f'
                    %(epoch+1, epochs, i+1, len(train_list)//batch_size, running_loss / log_interval, d_loss, loss_dur, loss_ce, loss_norm_rec, loss_F0_rec, loss_lm, loss_gen_all, loss_sty, loss_diff, d_loss_slm, loss_gen_lm, loss_s2s, loss_mono))
                
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
                    'train/s2s_loss': loss_s2s,
                    'train/mono_loss': loss_mono,
                    'iteration': iters
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
                    texts, bert_texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

                    if mels.size(-1) < 80:
                        continue

                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2 ** n_down)).to('cuda')
                        text_mask = length_to_mask(input_lengths).to(texts.device)

                        s2s_pred, s2s_attn = perform_text_alignment(model, mels, mask, texts)

                        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        # encode
                        t_en = model.text_encoder(texts, input_lengths, text_mask)
                        asr = (t_en @ s2s_attn_mono)

                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    ss = []
                    gs = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item())
                        mel = mels[bib, :, :mel_input_length[bib]]
                        s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                        ss.append(s)
                        s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                        gs.append(s)

                    # Fix: Use squeeze(1) instead of squeeze() to avoid removing batch dimension
                    s = torch.stack(ss); s = s.squeeze(tuple(range(1, len(s.shape))))
                    gs = torch.stack(gs); gs = gs.squeeze(tuple(range(1, len(gs.shape))))
                    s_trg = torch.cat([s, gs], dim=-1).detach()

                    bert_dur = model.bert(bert_texts, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
                    d, p = model.predictor(d_en, s, 
                                                        input_lengths, 
                                                        s2s_attn_mono, 
                                                        text_mask)
                    # get clips
                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en = []
                    gt = []

                    p_en = []
                    wav = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)

                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[bib, :, random_start:random_start+mel_len])
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

                    _, loss_dur = calculate_duration_and_ce_losses(d, d_gt, input_lengths)
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
        
        # Log metrics to wandb
        wandb.log({
            "eval/mel_loss": loss_test / iters_test,
            "eval/dur_loss": loss_align / iters_test,
            "eval/F0_loss": loss_f / iters_test,
            "epoch": epoch + 1
        })

        if (epoch + 1) % save_freq == 0 :
            if (loss_test / iters_test) < best_loss:
                best_loss = loss_test / iters_test
            state = {
                'net':  {key: model[key].state_dict() for key in model}, 
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': loss_test / iters_test,
                'epoch': epoch,
            }
            save_path = osp.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
            print(f'Saving to {save_path}')
            torch.save(state, save_path)

            # if estimate sigma, save the estimated simga
            if model_params.diffusion.dist.estimate_sigma_data:
                config['model_params']['diffusion']['dist']['sigma_data'] = float(np.mean(running_std))

                with open(osp.join(log_dir, osp.basename(config_path)), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)

                            
if __name__=="__main__":
    main()
