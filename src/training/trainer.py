import torch
import logging
import time

from src.training.loss import LossCalibrator
from src.models.registry import save_weights

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        # Clears all per-run accumulators so the same instance can be called
        # multiple times without metrics from a previous run contaminating the next.
        self.train_losses, self.val_losses, self.track_tokens_seen = [], [], []
        # Supplementary training health metrics recorded at each eval step
        self.learning_rates, self.grad_norms, self.peak_memory_gb = [], [], []
        self.tokens_seen, self.global_step = 0, -1
        # Tracks the best validation loss seen so far for best-model checkpointing
        self._best_val_loss = float("inf")
        self.step_times_sec = []
        self._step_start_time = None 

    def __call__(self, model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer,
                       gradient_accumulation_steps=1, scheduler=None, model_name=None,
                       lora: bool = False):

        self._reset_state()

        # Main training loop
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode

            for i, (input_batch, target_batch) in enumerate(train_loader):
                loss = LossCalibrator.calc_loss_batch(input_batch, target_batch, model, device)
                # Scale loss so accumulated gradients match a single full-batch update
                loss = loss / gradient_accumulation_steps
                loss.backward() # Calculate loss gradients
                self.tokens_seen += input_batch.numel() # Returns the total number of elements (or tokens) in the input_batch.

                is_last_batch = (i + 1) == len(train_loader)
                if (i + 1) % gradient_accumulation_steps == 0 or is_last_batch:
                    # Clip the L2 norm of all parameter gradients to max_norm=0.5.
                    # If the combined gradient vector is larger than 0.5, every gradient
                    # is scaled down proportionally so the norm equals 0.5.
                    # This prevents a single bad batch from producing exploding gradients
                    # that destabilise training — important for Gemma3 (random init) and
                    # also protects GPT-2 fine-tuning runs.
                    # clip_grad_norm_ returns the pre-clip norm, which we record at eval
                    # steps to diagnose instability (persistent norm == max_norm means
                    # clipping is always active and lr/init may need tuning).
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step() # Update model weights using loss gradients
                    # Advance the LR schedule by one step after each weight update.
                    # Called per optimizer step (not per epoch) so warmup and decay
                    # progress smoothly regardless of dataset size or batch size.
                    # scheduler=None means a fixed LR is used — behaviour is unchanged
                    # for callers that don't pass a scheduler.
                    if scheduler is not None:
                        scheduler.step()

                    # Record time for this optimizer step (pure training time, not including eval).
                    # The start time is reset AFTER eval below so that eval time is never
                    # absorbed into the next step's measurement.
                    if self._step_start_time is not None:
                        self.step_times_sec.append(time.time() - self._step_start_time)

                    optimizer.zero_grad(set_to_none=True) # Reset loss gradients from previous batch iteration (set_to_none frees memory immediately)
                    # Free MPS/CUDA cache after each weight update to avoid fragmentation
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    self.global_step += 1
                    # Optional evaluation step
                    if self.global_step % eval_freq == 0:
                        train_loss, val_loss = ModelTrainer.evaluate_model(
                            model, train_loader, val_loader, device, eval_iter)
                        self.train_losses.append(train_loss)
                        self.val_losses.append(val_loss)
                        self.track_tokens_seen.append(self.tokens_seen)

                        # LR: read from scheduler if present, otherwise from optimizer
                        lr = (scheduler.get_last_lr()[0] if scheduler is not None
                              else optimizer.param_groups[0]["lr"])
                        self.learning_rates.append(lr)

                        # Pre-clip gradient norm captured from clip_grad_norm_ above
                        self.grad_norms.append(grad_norm.item())

                        # Peak memory in GB — CUDA resets the high-water mark after each
                        # sample so we see per-interval peaks; MPS reports current usage
                        self.peak_memory_gb.append(
                            ModelTrainer._peak_memory_gb(device)
                        )
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)

                        logger.info(
                            f"[Evaluation] Epoch {epoch+1} (Step {self.global_step:06d}): "
                            f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f} | "
                            f"LR {lr:.2e}, Grad norm {grad_norm.item():.3f}, "
                            f"Mem {self.peak_memory_gb[-1]:.2f} GB"
                        )
                        # Save a checkpoint whenever val loss improves.
                        # This guards against overfitting: if val loss dips then rises,
                        # the saved weights are from the best generalising point, not
                        # the end of training. Skipped if model_name is not provided.
                        if model_name is not None and val_loss < self._best_val_loss:
                            self._best_val_loss = val_loss
                            save_weights(model, model_name, lora=lora)
                            logger.info(f"[Checkpoint] New best val loss {val_loss:.3f} — weights saved")

                    # Reset timer AFTER eval so eval time is excluded from the next step.
                    self._step_start_time = time.time()


            # Flush unused cached memory after each epoch to prevent OOM on MPS/CUDA
            if device.type == "mps":
                torch.mps.empty_cache()

            # Print a sample text after each epoch
            ModelTrainer.generate_and_print_sample(
                model, tokenizer, device, start_context
            )
            # Reset timer after sample generation so the first step of the next
            # epoch doesn't absorb text-generation time into its measurement.
            self._step_start_time = time.time()

        return self.train_losses, self.val_losses, self.track_tokens_seen
    
    @staticmethod
    def _peak_memory_gb(device) -> float:
        if device.type == "cuda":
            return torch.cuda.max_memory_allocated(device) / 1e9
        if device.type == "mps":
            return torch.mps.driver_allocated_memory() / 1e9
        return 0.0

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



