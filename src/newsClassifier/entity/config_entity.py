from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path = "artifacts\\data_ingestion"
    source_URL: str = "https://drive.google.com/file/d/14_s93gevc_8b6rPYBag4wAF9KhljBjzr/view?usp=sharing"
    local_data_file: Path = "artifacts\\data_ingestion\\data.zip"
    unzip_dir: Path = "artifacts\\data_ingestion"

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    base_model_path: Path = "artifacts\\prepare_base_model"
    model_name = "new_classification_model.pth"
    max_sequence_length: int = 128
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 8
    ffn_hidden_layers: int = 3072
    vocab_size: int = 30522
    batch_size: int = 32
    dropout: float = 0.2
    learning_rate: float = 2e-5
    weight_decay: float = 1e-12
    number_training_epochs: int = 1

@dataclass(frozen=True)
class TrainingDataPreprationConfig:
    input_data_path: Path = "artifacts\\data_ingestion\\data.csv"
    tokenizer_path: Path = "artifacts\\prepare_base_model\\tokenizer"
    tokenizer_model_id = "bert-base-cased"
    train_test_split: float = 0.20
