import torch
import logging

from src.training.loss import LossCalibrator

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        # Initialize lists to track losses and tokens seen
        self.train_losses, self.val_losses, self.track_tokens_seen = [], [], []
        self.tokens_seen, self.global_step = 0, -1

    def __call__(self, model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
        
        # Main training loop
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad() # Reset loss gradients from previous batch iteration
                loss = LossCalibrator.calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward() # Calculate loss gradients
                optimizer.step() # Update model weights using loss gradients
                self.tokens_seen += input_batch.numel() # Returns the total number of elements (or tokens) in the input_batch.
                self.global_step += 1
                # Optional evaluation step
                if self.global_step % eval_freq == 0: 
                    train_loss, val_loss = ModelTrainer.evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)
                    self.track_tokens_seen.append(self.tokens_seen)
                    logger.info(f"[Evaluation] Epoch {epoch+1} (Step {self.global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")


            # Print a sample text after each epoch
            ModelTrainer.generate_and_print_sample(
                model, tokenizer, device, start_context
            )

        return self.train_losses, self.val_losses, self.track_tokens_seen
    
    @staticmethod
    def evaluate_model(model, train_loader, val_loader, device, eval_iter):
        model.eval()
        with torch.no_grad():
            train_loss = LossCalibrator.calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            val_loss = LossCalibrator.calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()
        return train_loss, val_loss
    
    @staticmethod
    def generate_and_print_sample(model, tokenizer, device, start_context):
        model.eval()
        context_size = model.get_context_size()
        encoded = ModelTrainer.text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            token_ids = ModelTrainer.generate_text_simple(
                model=model, idx=encoded,
                max_new_tokens=50, context_size=context_size
            )
        decoded_text = ModelTrainer.token_ids_to_text(token_ids, tokenizer)
        clean_text = decoded_text.replace("\n", " ")
        logger.info(f"Sample :{clean_text}") # Compact print format
        model.train()

    
    @staticmethod
    def text_to_token_ids(text, tokenizer):
        encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
        return encoded_tensor
    
    @staticmethod
    def token_ids_to_text(token_ids, tokenizer):
        flat = token_ids.squeeze(0) # remove batch dimension
        return tokenizer.decode(flat.tolist())
    
    @staticmethod
    def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

        # For-loop is the same as before: Get logits, and only focus on last time step
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]

            # New: Filter logits with top_k sampling
            if top_k is not None:
                # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)

            # New: Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # Otherwise same as before: get idx of the vocab entry with the highest logits value
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            if eos_id is not None and (idx_next == eos_id).any():  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                break

            # Same as before: append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

        return idx



