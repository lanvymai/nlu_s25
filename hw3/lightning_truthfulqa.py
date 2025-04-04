import os
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TruthfulQAModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

if __name__ == "__main__":
    # Initialize the model
    model_name = "facebook/opt-6.7b"
    model = TruthfulQAModel(model_name)

    # Configure the Trainer
    trainer = Trainer(
        accelerator="gpu",  # Use GPU
        devices=1,          # Number of GPUs per node
        strategy=DDPStrategy(find_unused_parameters=False),  # Distributed training
        max_epochs=1,       # Number of epochs
        log_every_n_steps=10,
    )

    # Train the model (replace with your actual training logic)
    rank_zero_info("Starting training...")
    trainer.fit(model)