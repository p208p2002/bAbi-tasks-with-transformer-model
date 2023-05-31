from datasets import load_dataset   
from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

INPUT_TEMPLATE = """
Context:
{context}

Question:
{question}

Answer:
{answer}
"""

def get_next_qa(dataset):
    for x in dataset:
        story = x.get('story', None)
        if story:
            sentences = story.get("text", None)
            sent_types = story.get("type",[])
            
            context = ""
            for s_idx, sent in enumerate(sentences):
                if sent_types[s_idx] == 1:
                    question = sent
                    answer = story["answer"][s_idx]
                    context = context.strip()
                    yield context, question, answer
                else:
                    context += f"{sent}\n"


class BabiqaDataset():
    def __init__(self,tokenizer,task_no="qa1",split="train",no_answer=False,retrun_object=False) -> None:
        self.tokenizer:PreTrainedTokenizer = tokenizer
        dataset = load_dataset('babi_qa', type='en', task_no=task_no)[split]
        self.data = list(get_next_qa(dataset))
        self.no_answer = no_answer
        self.retrun_object = retrun_object
    
    def __getitem__(self,index):
        context, question, answer = self.data[index]
        cqa = {
            "context":context,
            "question":question,
            "answer":answer
        }
        
        
        if self.retrun_object:
            return cqa

        if self.no_answer:
            cqa["answer"] = ""
        
        
        input_text = INPUT_TEMPLATE.format_map(cqa).strip()
        encodings = self.tokenizer(input_text,truncation=True,max_length=384,return_tensors="pt")
        encodings["labels"] = encodings["input_ids"].clone()
        return {
            "input_ids":encodings["input_ids"],
            "labels":encodings["labels"]
        }
    
    def __len__(self):
        return len(self.data)


def collate_data(batch,padding_value,label_padding_value=-100):
    new_batch = defaultdict(lambda:[])
    for x in batch:
        for x_key in x.keys():
            new_batch[x_key].append(x[x_key][0])
    
    new_batch = dict(new_batch)
    for batch_key in new_batch.keys():
        if batch_key == "labels":
            new_batch[batch_key] = pad_sequence(new_batch[batch_key],batch_first=True,padding_value=label_padding_value)
        else:
            new_batch[batch_key] = pad_sequence(new_batch[batch_key],batch_first=True,padding_value=padding_value)
    
    return new_batch

