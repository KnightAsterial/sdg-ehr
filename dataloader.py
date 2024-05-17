# using Hugging Face loads in data 


# DataLoader instance for training and evaluation datasets
from torch.utils.data import DataLoader

# Create instances of the NarrativeDataset
train_dataset = NarrativeDataset(tokenized_train_dataset)
eval_dataset = NarrativeDataset(tokenized_eval_dataset)

# Define data collator
def data_collator(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_masks = torch.stack([item[1] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_masks}

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=data_collator)
eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=False, collate_fn=data_collator)
