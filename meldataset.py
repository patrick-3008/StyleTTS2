# coding: utf-8
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
import logging
import os # Import os module for path joining

# Set up logging
logger = logging.getLogger(__name__)
# Set level higher than DEBUG for less verbose output during normal runs
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG) # Keep DEBUG for now if you want detailed logs

from char_indexer import VanillaCharacterIndexer, BertCharacterIndexer

# Set random seeds for reproducibility
np.random.seed(1)
random.seed(1)

# Mel Spectrogram transformation parameters
# Assuming 24000 Hz sample rate based on your assertion
to_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=24000, # Explicitly set sample rate
    n_mels=80,
    n_fft=2048,
    win_length=1200,
    hop_length=300
)
mean, std = -4, 4

def preprocess(wave, sample_rate=24000):
    # Ensure waveform is a torch tensor and float
    if isinstance(wave, np.ndarray):
        wave_tensor = torch.from_numpy(wave).float()
    elif isinstance(wave, torch.Tensor):
        wave_tensor = wave.float()
    else:
        logger.error(f"Expected numpy array or torch tensor, but got {type(wave)}")
        raise TypeError(f"Expected numpy array or torch tensor, but got {type(wave)}")

    # Resample if necessary (should be handled in __init__, but safety check)
    if sample_rate != 24000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)
        wave_tensor = resampler(wave_tensor)
        # sample_rate = 24000 # Update sample rate if you wanted to track it here

    # Ensure waveform is mono (remove channel dimension if multi-channel)
    if wave_tensor.ndim > 1 and wave_tensor.shape[0] > 1:
        # Assuming multi-channel, take the first channel
        wave_tensor = wave_tensor[0, :]
    elif wave_tensor.ndim == 1:
        pass # Already mono
    else:
        logger.warning(f"Unexpected waveform shape after loading/resampling: {wave_tensor.shape}. Attempting to process as 1D.")
        wave_tensor = wave_tensor.squeeze() # Try to remove any singleton dimensions

    # Apply mel spectrogram transform
    # to_mel expects a 1D tensor (time,) or 2D (batch, time). Since we process one sample at a time, 1D is expected.
    if wave_tensor.ndim != 1:
        logger.error(f"Waveform tensor is not 1D after processing, shape: {wave_tensor.shape}")
        raise ValueError(f"Waveform tensor is not 1D for mel spectrogram.")

    mel_tensor = to_mel(wave_tensor)

    # Apply normalization
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std # Add batch dim for normalization

    return mel_tensor


class FilePathDataset(torch.utils.data.Dataset):
    # Added data_wavs_dir parameter to __init__
    def __init__(self, dataset_list, data_wavs_dir, data_augmentation=False, validation=False, min_length=50):
        # dataset_list is expected to be a list of strings like "filename.wav|text"
        # data_wavs_dir is the path to the directory containing the .wav files

        self.data_wavs_dir = data_wavs_dir # Store the wavs directory path
        self.char_indexer = VanillaCharacterIndexer()
        self.bert_char_indexer = BertCharacterIndexer()

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        self.min_length = min_length

        # --- Modification to __init__ Starts Here ---
        self._processed_dataset = [] # List to store sample dictionaries
        skipped_files = 0

        logger.info(f"Processing dataset with {len(dataset_list)} entries from metadata...")

        # Iterate through the input list of "filename|text" strings
        # Based on your data prep script, dataset_list comes from reading the CSV
        # and the CSV now has format filename|text|sampling_rate
        for i, line in enumerate(dataset_list):
            try:
                parts = line.strip().split('|')

                # Expecting 3 parts now: filename, text, sampling_rate
                if len(parts) != 3:
                    logger.warning(f"Skipping malformed line {i}: {line.strip()}. Expected 3 parts, found {len(parts)}")
                    skipped_files += 1
                    continue

                filename_wav, text, sampling_rate_str = parts
                audiopath = os.path.join(self.data_wavs_dir, filename_wav)

                # Convert sampling rate string to integer
                try:
                    sample_rate = int(sampling_rate_str)
                except ValueError:
                    logger.warning(f"Skipping line {i} with invalid sampling rate format: {line.strip()}")
                    skipped_files += 1
                    continue


                # Load audio waveform - __getitem__ expects 'audio' key with waveform
                try:
                    # Load audio at its original sampling rate first
                    waveform, original_sample_rate = torchaudio.load(audiopath)

                    # Resample if necessary to 24000 Hz as expected by preprocess and assertion
                    if original_sample_rate != 24000:
                        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=24000)
                        waveform = resampler(waveform)
                        # The 'sample_rate' stored in the dict should match the *processed* rate
                        processed_sample_rate = 24000
                    else:
                        processed_sample_rate = original_sample_rate # Should be 24000 if already correct


                    # Basic check for silent files or very short audio
                    if waveform.numel() == 0 or waveform.abs().max() < 1e-6:
                        logger.warning(f"Skipping empty or silent audio file: {audiopath}")
                        skipped_files += 1
                        continue # Skip this sample


                except Exception as e:
                    logger.warning(f"Skipping {audiopath} due to error loading or resampling: {e}")
                    skipped_files += 1
                    continue # Skip this sample if audio loading fails

                # Create the sample dictionary expected by __getitem__ and _load_tensor
                # Note: This assumes the 'text' from the CSV is the 'phonemes' representation
                # If your text is raw characters, you might need a phonemizer here before storing in 'phonemes'
                sample_data = {
                    'filename_wav': filename_wav,
                    'text': text,     # Original text from CSV
                    'phonemes': text, # Assuming 'text' is the phoneme representation needed by _load_tensor
                    # Store the loaded waveform tensor (ensure it's 1D or 2D as expected later)
                    # _load_tensor expects sample['audio'][0] to be a numpy array
                    # Let's store the waveform tensor and convert to numpy array later in _load_tensor
                    'audio': waveform, # Store the loaded waveform tensor (might be 1D or 2D)
                    'sampling_rate': processed_sample_rate, # Store the actual (or resampled) sampling rate (should be 24000)
                    'speaker_id': 0 # Assuming a single speaker or speaker_id is not in CSV
                }

                # Append the dictionary to the processed dataset list
                self._processed_dataset.append(sample_data)

            except Exception as e:
                logger.error(f"Unexpected error processing line {i}: {line.strip()} - {e}")
                skipped_files += 1


        # Assign the processed list of dictionaries to self.dataset
        self.dataset = self._processed_dataset

        logger.info(f"Successfully loaded {len(self.dataset)} samples. Skipped {skipped_files} files.")
        if not self.dataset:
            logger.error("No valid samples loaded into the dataset. Please check your metadata file and audio paths.")
            # Consider raising an error or exiting if no data is loaded
            raise RuntimeError("No data loaded into the dataset.") # Raise an error to stop execution


    def __len__(self):
        # Length is now the number of processed samples
        return len(self.dataset)

    def __getitem__(self, idx):
        # sample is now guaranteed to be a dictionary from __init__
        sample = self.dataset[idx]
        logger.debug(f"DEBUG: Processing sample {idx}: {sample.get('filename_wav', 'N/A')}")

        # Check if the sample contains the necessary keys (redundant if init is correct, but good safety)
        # This check should now pass if __init__ successfully created the dictionary with these keys
        required_keys = ['sampling_rate', 'phonemes', 'audio']
        for key in required_keys:
            if key not in sample:
                logger.error(f"Missing required key '{key}' in sample {idx}")
                raise KeyError(f"Missing required key '{key}' in sample {idx}") # Error previously happened here

        # Validate sample's sampling rate (This assertion should now pass if __init__ is correct)
        # It's good practice to keep this assertion for safety
        assert sample['sampling_rate'] == 24000, f"Invalid sampling rate {sample['sampling_rate']} for sample {idx}"

        # Load original sample data from the dictionary
        # _load_tensor now receives a dictionary with 'phonemes', 'audio', 'speaker_id'
        wave, text_tensor, bert_text_tensor, speaker_id = self._load_tensor(sample)

        # Validate mel tensor shape
        # Pass sampling rate to preprocess if needed, though preprocess is hardcoded to 24k now
        mel_tensor = preprocess(wave, sample_rate=sample['sampling_rate']).squeeze() # Pass actual sample rate to preprocess
        if mel_tensor.size(1) == 0:
            logger.warning(f"Empty mel tensor for sample {idx} ({sample.get('filename_wav', 'N/A')}). Skipping.")
            # Returning None here would require the Collater to handle None values
            # For now, let's proceed, but be aware of potential issues if mel_tensor is empty

        # Process original audio into mel spectrogram
        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        # Ensure length_feature is positive before modulo
        if length_feature > 0:
            acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
        else:
            logger.warning(f"Acoustic feature length is zero for sample {idx} ({sample.get('filename_wav', 'N/A')}). Skipping padding adjustment.")
            # You might need more robust handling for zero-length features


        # Get reference sample from the same speaker
        # This still randomly samples from the *entire* processed dataset
        # If you have multiple speakers and need same-speaker refs, this logic needs refinement
        ref_idx = random.randint(0, len(self.dataset) - 1)
        # Ensure the randomly selected reference sample is also valid
        try:
            same_speaker_sample = self.dataset[ref_idx] # Gets a dictionary
            ref_mel_tensor, _ = self._load_data(same_speaker_sample) # _load_data expects a dictionary
        except Exception as e:
            logger.warning(f"Could not load reference sample {ref_idx}. Error: {e}. Attempting to use current sample as ref.")
            # Fallback: use the current sample as reference if random one fails
            ref_mel_tensor, _ = self._load_data(sample) # Use current sample as ref


        return speaker_id, acoustic_feature, text_tensor, bert_text_tensor, ref_mel_tensor, wave

    def _load_tensor(self, sample):
        # This method now correctly receives a 'sample' dictionary with expected keys
        text = sample['phonemes'] # Gets phonemes from the dictionary
        speaker_id = sample.get('speaker_id', 0) # Gets speaker_id from the dictionary

        # Ensure the audio data is valid and is a torch tensor
        # The 'audio' key now holds the loaded waveform tensor from __init__
        wave_tensor = sample['audio']

        if wave_tensor is None or wave_tensor.numel() == 0: # Check for empty tensor
            logger.error(f"Empty audio data tensor in sample {sample.get('filename_wav', 'N/A')}")
            raise ValueError(f"Empty audio data tensor in sample {sample.get('filename_wav', 'N/A')}")

        # --- Modification Starts Here ---
        # Ensure waveform is mono (remove channel dimension if multi-channel)
        if wave_tensor.ndim > 1 and wave_tensor.shape[0] > 1:
            # Assuming multi-channel, take the first channel
            logger.debug(f"Converting multi-channel audio {sample.get('filename_wav', 'N/A')} to mono. Shape: {wave_tensor.shape}")
            wave_tensor = wave_tensor[0, :]
        elif wave_tensor.ndim == 1:
            # Already mono
            pass
        else:
            logger.warning(f"Unexpected waveform shape before NumPy conversion: {wave_tensor.shape}. Attempting to squeeze.")
            wave_tensor = wave_tensor.squeeze() # Try to remove any singleton dimensions
            if wave_tensor.ndim != 1:
                logger.error(f"Waveform tensor is not 1D after squeezing, shape: {wave_tensor.shape}")
                raise ValueError(f"Waveform tensor is not 1D for padding.")

        # Ensure waveform is on CPU before converting to numpy for padding
        if wave_tensor.is_cuda:
            wave_tensor = wave_tensor.cpu()

        wave_np = wave_tensor.numpy() # Convert tensor to numpy (should now be 1D)

        # Apply padding
        # Ensure padding zeros match the numpy array dtype
        # This should now work as wave_np is 1D
        wave_padded_np = np.concatenate([np.zeros([5000], dtype=wave_np.dtype), wave_np, np.zeros([5000], dtype=wave_np.dtype)], axis=0)
        wave_padded_tensor = torch.from_numpy(wave_padded_np).float() # Convert back to float tensor
        # --- Modification Ends Here ---


        # Apply indexers to the text (phonemes)
        char_idx = self.char_indexer(text)
        bert_char_idx = self.bert_char_indexer(text)

        # Add start and end tokens (assuming 0 is the pad/end token)
        char_idx.insert(0, 0); bert_char_idx.insert(0, 0)
        char_idx.append(0); bert_char_idx.append(0)

        char_idx = torch.LongTensor(char_idx)
        bert_char_idx = torch.LongTensor(bert_char_idx)

        return wave_padded_tensor, char_idx, bert_char_idx, speaker_id

    def _load_data(self, data):
        # data is now a dictionary, passed from __getitem__
        # Call _load_tensor with the dictionary
        wave_padded_tensor, _, _, speaker_id = self._load_tensor(data) # _load_tensor expects a dict

        # Preprocess the padded waveform to get mel spectrogram
        # Pass the correct sample rate to preprocess
        mel_tensor = preprocess(wave_padded_tensor, sample_rate=data['sampling_rate']).squeeze() # Pass actual sample rate


        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            # Ensure random_start is valid
            max_start = mel_length - self.max_mel_length
            if max_start > 0:
                random_start = np.random.randint(0, max_start + 1) # +1 to include the last possible start
            else:
                random_start = 0 # Sequence is too short or exactly max_mel_length

            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]
        elif mel_length < self.max_mel_length: # Check against max_mel_length for consistency with padding
            # Collater will handle padding for sequences shorter than max_mel_length
            logger.debug(f"Mel tensor length {mel_length} is less than max_mel_length {self.max_mel_length} (padding expected)")
            pass # Collater will handle padding
        # If mel_length == self.max_mel_length, no slicing needed

        return mel_tensor, speaker_id

# Update build_dataloader to pass data_wavs_dir
# This assumes dataset_list is the list of strings read from the metadata CSV
def build_dataloader(dataset_list, data_wavs_dir, min_length, batch_size, num_workers, device, validation=False, collate_config={}, dataset_config={}, **kwargs):
    # dataset_list is the list of "filename|text|sampling_rate" strings read from metadata
    dataset = FilePathDataset(dataset_list, data_wavs_dir=data_wavs_dir, min_length=min_length, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader

# The Collater class remains the same as provided previously.
# (Keeping it here for completeness, but no changes were needed for the KeyError)
class Collater(object):
    """
    Collates the batch of data, ensuring proper padding and sorting.
    """

    def __init__(self):
        self.text_pad_index = 0
        self.min_mel_length = 192 # Check if this should be self.max_mel_length or another config
        self.max_mel_length = 192

    def __call__(self, batch):
        # Filter out None samples if you add logic to skip samples in __getitem__
        # batch = [b for b in batch if b is not None]
        # if not batch:
        #     return None # Or handle appropriately

        # Sort by mel length
        # Ensure element b[1] (acoustic_feature/mel_tensor) is valid
        try:
            lengths = [b[1].shape[1] for b in batch]
        except AttributeError as e:
            logger.error(f"Error getting mel length in Collater: {e}. Batch item: {batch}")
            raise

        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch]) # Assuming text_tensor is b[2]

        labels = torch.zeros((len(batch))).long()
        mels = torch.zeros((len(batch), nmels, max_mel_length)).float()
        texts = torch.zeros((len(batch), max_text_length)).long()
        bert_texts = torch.zeros((len(batch), max_text_length)).long()

        input_lengths = torch.zeros(len(batch)).long()
        output_lengths = torch.zeros(len(batch)).long()
        ref_mels = torch.zeros((len(batch), nmels, self.max_mel_length)).float() # Ref mel padding uses max_mel_length from Collater init
        waves = [None for _ in range(len(batch))] # Store raw waves if needed

        for bid, (label, mel, text, bert_text, ref_mel, wave) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)

            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            bert_texts[bid, :text_size] = bert_text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size

            ref_mel_size = ref_mel.size(1)
            # Pad ref_mels to self.max_mel_length from Collater init
            if ref_mel_size > self.max_mel_length:
                # Truncate if reference mel is longer than the max allowed in Collater
                ref_mels[bid, :, :] = ref_mel[:, :self.max_mel_length]
            elif ref_mel_size > 0:
                ref_mels[bid, :, :ref_mel_size] = ref_mel
            else:
                logger.warning(f"Reference mel tensor size is zero for batch item {bid}. Skipping padding.")


            # Optionally store the raw wave
            waves[bid] = wave # Keep this line if you need the raw wave in the batch

        return waves, texts, bert_texts, input_lengths, mels, output_lengths, ref_mels

# Example usage (This part is just for illustration, not part of the MelDataset class itself)
if __name__ == '__main__':
    # This is how you would use the updated class, assuming you have
    # a list of metadata lines and the path to your wav files.

    # Example metadata lines (replace with loading from your actual CSV)
    # Assume the CSV now has filename|text|sampling_rate based on previous script edit
    metadata_lines_example = [
        "audio_file_001.wav|This is the first sentence.|24000",
        "audio_file_002.wav|This is the second sentence for testing.|24000",
        # Add more lines as needed
    ]

    # Replace with the actual path to your wav directory
    # This should be the same path as target_wav_dir in your data prep script
    wav_directory = '/content/data/wavs' # Example path

    # --- How to read your actual metadata CSV ---
    actual_metadata_path = '/content/data/metadata_train.csv' # Replace with your path
    dataset_list_from_csv = []
    if os.path.exists(actual_metadata_path):
        with open(actual_metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset_list_from_csv.append(line.strip())
    else:
        print(f"Warning: Metadata file not found at {actual_metadata_path}. Using example data.")
        dataset_list_from_csv = metadata_lines_example # Fallback to example if file not found


    # Create an instance of the dataset
    try:
        training_dataset = FilePathDataset(
            dataset_list=dataset_list_from_csv, # Pass the list read from your CSV
            data_wavs_dir=wav_directory, # Pass the wavs directory here
            validation=False,
            min_length=50
        )

        # Create a dataloader
        training_dataloader = build_dataloader(
            dataset_list=dataset_list_from_csv, # Pass the same list here
            data_wavs_dir=wav_directory, # Pass the wavs directory here
            min_length=50,
            batch_size=4, # Example batch size
            num_workers=2, # Example num workers
            device='cpu' # Or 'cuda' if using GPU
        )

        print(f"Dataset size: {len(training_dataset)}")
        # Iterate through the dataloader to test
        for i, batch in enumerate(training_dataloader):
            print(f"Batch {i+1}:")
            # Unpack the batch tuple based on Collater's return
            waves, texts, bert_texts, input_lengths, mels, output_lengths, ref_mels = batch

            print(f"  Waves (list of tensors): {len(waves)} items")
            print(f"  Texts (tensor): {texts.shape}")
            print(f"  BERT Texts (tensor): {bert_texts.shape}")
            print(f"  Input Lengths (tensor): {input_lengths.shape}")
            print(f"  Mels (tensor): {mels.shape}")
            print(f"  Output Lengths (tensor): {output_lengths.shape}")
            print(f"  Ref Mels (tensor): {ref_mels.shape}")
            # You can add more checks here, e.g., assert shapes match expectations

            if i > 2: # Process a few batches
                break

    except Exception as e:
        logger.error(f"An error occurred during dataloader creation or iteration: {e}")
