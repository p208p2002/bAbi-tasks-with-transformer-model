from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer as DefaultTrainer
from data import collate_data, BabiqaDataset
from torch.utils.data import ConcatDataset
import torch
from transformers.optimization import get_scheduler
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_name_or_id")
parser.add_argument('-lr',default=3e-4,type=float)
parser.add_argument('-batch_size',default=6,type=int)
parser.add_argument('-epoch',default=3,type=int)
parser.add_argument('-ga','--gradient_accumulation',default=1,type=int)
args = parser.parse_args()

class Trainer(DefaultTrainer):
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        disable scheduler
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=0,
                num_training_steps=sys.maxsize,
            )
        return self.lr_scheduler

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_id)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_id)

train_dataset = ConcatDataset(
    [
        BabiqaDataset(tokenizer, split="train", task_no=f"qa{task_id+1}")
        for task_id in range(20)
    ]
)
test_dataset = ConcatDataset(
    [
        BabiqaDataset(tokenizer, split="test", task_no=f"qa{task_id+1}")
        for task_id in range(20)
    ]
)

training_args = TrainingArguments(
    output_dir="my_model",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=args.lr,
    num_train_epochs=args.epoch,
    weight_decay=0.0,
    push_to_hub=False,
    load_best_model_at_end=True,
    per_gpu_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=lambda x: collate_data(x, padding_value=tokenizer.eos_token_id,label_padding_value=tokenizer.eos_token_id),
)

trainer.train()
trainer.save_model("my_model/best")
tokenizer.save_pretrained("my_model/best")
