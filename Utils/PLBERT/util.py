import os
import yaml
import torch
from transformers import AlbertConfig, AlbertModel

from char_indexer import BertCharacterIndexer

symbols = BertCharacterIndexer.symbols

from huggingface_hub import hf_hub_download

class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state
    
def _load_plbert(config_path, model_path):
    plbert_config = yaml.safe_load(open(config_path))
    
    albert_base_configuration = AlbertConfig(**plbert_config['model_params'], vocab_size=len(symbols))
    bert = CustomAlbert(albert_base_configuration)

    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['net']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        if name.startswith('encoder.'):
            name = name.replace('encoder.', '')
            new_state_dict[name] = v
    if "embeddings.position_ids" in new_state_dict: del new_state_dict["embeddings.position_ids"]
    bert.load_state_dict(new_state_dict, strict=False)
    
    return bert

def load_plbert(repo_id, dirname):
    config_path = hf_hub_download(
        repo_id=repo_id,       # or e.g. "stabilityai/stable-diffusion-2"
        filename=f"{dirname}/config.yml",
        repo_type="model",                       # or "dataset", or "space"
    )
    
    model_path = hf_hub_download(
        repo_id=repo_id,       # or e.g. "stabilityai/stable-diffusion-2"
        filename=f"{dirname}/model.pth",
        repo_type="model",                       # or "dataset", or "space"
    )

    return _load_plbert(config_path, model_path)