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
        # Clear accumulators so the same instance can be reused across runs.
        self.train_losses, self.val_losses, self.track_tokens_seen = [], [], []
        self.learning_rates, self.grad_norms, self.peak_memory_gb = [], [], []
        self.tokens_seen, self.global_step = 0, -1
        self._best_val_loss = float("inf")
        self.step_times_sec = []
        self._step_start_time = None 

    def __call__(self, model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer,
                       gradient_accumulation_steps=1, scheduler=None, model_name=None,
                       lora: bool = False, grad_clip: float = 0.5):

        self._reset_state()

        for epoch in range(num_epochs):
            model.train()

            for i, (input_batch, target_batch) in enumerate(train_loader):
                loss = LossCalibrator.calc_loss_batch(input_batch, target_batch, model, device)
                # Scale so accumulated gradients equal a single full-batch update.
                loss = loss / gradient_accumulation_steps
                loss.backward()
                self.tokens_seen += input_batch.numel()

                is_last_batch = (i + 1) == len(train_loader)
                if (i + 1) % gradient_accumulation_steps == 0 or is_last_batch:
                    # clip_grad_norm_ returns pre-clip norm — recorded to diagnose instability.
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    optimizer.step()
                    # Step per optimizer update, not per epoch.
                    if scheduler is not None:
                        scheduler.step()

                    # Pure training time — timer resets after eval so eval time is excluded.
                    if self._step_start_time is not None:
                        self.step_times_sec.append(time.time() - self._step_start_time)

                    optimizer.zero_grad(set_to_none=True)
                    # Free MPS cache after each update to prevent fragmentation.
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

                        # CUDA: peak since last reset; MPS: current usage.
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
                        # Save best checkpoint — guards against keeping overfit final weights.
                        if model_name is not None and val_loss < self._best_val_loss:
                            self._best_val_loss = val_loss
                            save_weights(model, model_name, lora=lora)
                            logger.info(f"[Checkpoint] New best val loss {val_loss:.3f} — weights saved")

                    # Reset timer AFTER eval so eval time is excluded from the next step.
                    self._step_start_time = time.time()


            # Flush unused cached memory after each epoch to prevent OOM on MPS/CUDA
            if device.type == "mps":
                torch.mps.empty_cache()

            # Sample generation
            ModelTrainer.generate_and_print_sample(
                model, tokenizer, device, start_context
            )
            # Exclude generation time from the first step of the next epoch.
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
        logger.info(f"Sample :{clean_text}")
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

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :]

            # top-k filtering
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)

            # Temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # Greedy decoding
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            if eos_id is not None and (idx_next == eos_id).any():
                break

            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

        return idx



