import torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoModelForMaskedLM, PreTrainedModel, RobertaConfig
import torch

class MeanPooler(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

class MultiTaskRoberta(nn.Module):
    def __init__(self, name="roberta-base", emb_dim=256, theme_path="top_themes.txt"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        self.pool = MeanPooler()
        hid = self.backbone.config.hidden_size

        # Heads
        self.proj = nn.Sequential(nn.Linear(hid, emb_dim), nn.GELU())
        num_themes = 0

        with open(theme_path) as f:
            num_themes = len(f.readlines())
        self.theme_head = nn.Linear(hid, num_themes)
        self.tone_head  = nn.Linear(hid, 2)

        # MLM – we reuse a tied-weights LM head by wrapping a separate module that shares the encoder
        self.mlm = AutoModelForMaskedLM.from_pretrained(name)
        # Tie encoder weights
        self.mlm.roberta = self.backbone

    def forward_embed(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(out.last_hidden_state, attention_mask)
        z = F.normalize(self.proj(pooled), p=2, dim=-1)
        return z, pooled  # z for triplet, pooled for themes/tone
    
    def save_checkpoint(self, path):
        """Save model checkpoint with config info"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'name': getattr(self, 'name', 'roberta-base'),
                'emb_dim': self.proj[0].out_features,
                'hidden_size': self.backbone.config.hidden_size,
                'vocab_size': self.backbone.config.vocab_size,
            }
        }
        torch.save(checkpoint, path)


    def forward_mlm(self, **batch):
        return self.mlm(**batch)

class BiasClassifier(PreTrainedModel):
    config_class = RobertaConfig

    def __init__(self, config, pretrained_path: str, num_bias_classes=3, freeze_backbone=False):
        super().__init__(config)
        
        if pretrained_path:
            # Load pretrained checkpoint
            with open(pretrained_path, 'rb') as f:
                checkpoint = torch.load(f, map_location='cpu')
           
            # Initialize backbone and pooler (reuse from pretrained)
            self.backbone = AutoModel.from_config(config)
            self.pool = MeanPooler()
            
            # Load pretrained backbone weights
            pretrained_state = checkpoint['model_state_dict']
            backbone_state = {k.replace('backbone.', ''): v for k, v in pretrained_state.items() 
                             if k.startswith('backbone.')}
            self.backbone.load_state_dict(backbone_state)
        else:
            raise ValueError("Pretrained path is required")
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # New bias classification head
        self.bias_head = nn.Linear(config.hidden_size, num_bias_classes)
        self.num_labels = num_bias_classes
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(out.last_hidden_state, attention_mask)
        logits = self.bias_head(pooled)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        # Return format expected by Trainer
        return {
            'loss': loss,
            'logits': logits,
        }
    
    @classmethod
    def from_pretrained_checkpoint(cls, pretrained_path, num_bias_classes=3, freeze_backbone=False):
        """Create model from your custom checkpoint"""
        config = RobertaConfig.from_pretrained("roberta-base")
        config.num_labels = num_bias_classes
        return cls(config, pretrained_path, num_bias_classes, freeze_backbone)
