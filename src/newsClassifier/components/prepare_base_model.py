import torch
import torch.nn as nn

from newsClassifier.entity.config_entity import PrepareBaseModelConfig
from newsClassifier.utils.common import get_device

from logger import logging

logger = logging.getLogger("BaseModelPrepration")

class DistilBertClassification(nn.Module):
  def __init__(
    self, 
    vocab_size:int, 
    d_model:int, 
    max_seq_length:int, 
    n_heads:int, 
    n_layers:int, 
    dropout:float, 
    ffn_hidden_layer:int
  ):
    super(DistilBertClassification, self).__init__()
    print(vocab_size)
    self.word_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=0)
    self.position_embeddings = nn.Embedding(max_seq_length, d_model)
    self.attention_layer = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
           d_model=d_model, 
           nhead=n_heads, 
           dropout=dropout, 
           batch_first=True, 
           dim_feedforward=ffn_hidden_layer
        )
        , num_layers=n_layers)
    self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
    self.classifier = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, 2)
    )
    self.dropout = nn.Dropout(dropout)

    self.initialize_weights()

  def initialize_weights(self):
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=1)

  def forward(self, X, input_mask=None):
    word_embeddings = self.word_embeddings(X)
    position_embeddings = self.position_embeddings(torch.arange(X.shape[1], device=X.device))
    embeddings = word_embeddings.to(get_device()) + position_embeddings.to(get_device())
    embeddings = self.dropout(embeddings)
    attention_output = self.attention_layer(embeddings)
    attention_output = self.dropout(attention_output)
    attention_output = self.layer_norm(attention_output)
    attention_output = self.dropout(attention_output)
    logits = self.classifier(attention_output)
    return logits

class PrepareBaseModel:
    def __init__(self) -> None:
        self.config = PrepareBaseModelConfig

    def get_base_model(self):
      model = DistilBertClassification(
        vocab_size=self.config.vocab_size,
        d_model=self.config.d_model,
        max_seq_length=self.config.max_sequence_length,
        n_heads=self.config.num_heads,
        n_layers=self.config.num_layers,
        dropout=self.config.dropout,
        ffn_hidden_layer=self.config.ffn_hidden_layers
      )

      return model