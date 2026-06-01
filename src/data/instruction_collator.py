import torch   

class InstructionCollator:
    def __init__(self, pad_token_id=50256, ignore_index=-100, allowed_max_length=None):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.allowed_max_length = allowed_max_length

    def __call__(self, batch):
        # padding + target masking logic
        # Find the longest sequence in the batch
        batch_max_length = max(len(item)+1 for item in batch)

        # Pad and prepare inputs and targets
        inputs_lst, targets_lst = [], []

        for item in batch:
            new_item = item.copy()
            # Add an <|endoftext|> token
            new_item += [self.pad_token_id]
            # Pad sequences to max_length
            padded = (
                new_item + [self.pad_token_id] *
                (batch_max_length - len(new_item))
            )
            inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
            targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

            # Replace all but the first padding tokens in targets by ignore_index
            mask = targets == self.pad_token_id
            indices = torch.nonzero(mask).squeeze(-1)
            if indices.numel() > 1:
                targets[indices[1:]] = self.ignore_index

            # Optionally truncate to maximum sequence length
            if self.allowed_max_length is not None:
                inputs = inputs[:self.allowed_max_length]
                targets = targets[:self.allowed_max_length]

            inputs_lst.append(inputs)
            targets_lst.append(targets)

        inputs_tensor = torch.stack(inputs_lst)
        targets_tensor = torch.stack(targets_lst)

        return inputs_tensor, targets_tensor