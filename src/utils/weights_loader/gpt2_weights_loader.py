import tensorflow
import torch
import numpy as np
import os
import json
import logging

from src.utils.gpt_download import download_and_load_gpt2, load_gpt2_params_from_tf_ckpt

from src.utils.weights_loader.base_weights_loader import BaseWeightsLoader

logger = logging.getLogger(__name__)

class GPT2WeightsLoader(BaseWeightsLoader):
    def __init__(self, model_size = "124M"):

        weights_path = "data/gpt2"
        model_path = os.path.join(weights_path, model_size)  

        if not os.path.exists(model_path):
            self.settings, self.params = download_and_load_gpt2(model_size=model_size, models_dir=weights_path)
            logger.info("[GPT2WeightsLoader] Successfully saved configuration and weights!")
        else:
            # Load settings and params
            tf_ckpt_path = tensorflow.train.latest_checkpoint(model_path)
            with open(os.path.join(model_path, "hparams.json"), "r", encoding="utf-8") as model_settings:
                self.settings = json.load(model_settings)
            self.params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, self.settings)
            logger.info("[GPT2WeightsLoader] Loading existing settings and parameters")
    
    def __call__(self, gpt):
        gpt.pos_emb.weight = GPT2WeightsLoader.assign(gpt.pos_emb.weight, self.params['wpe'])
        gpt.tok_emb.weight = GPT2WeightsLoader.assign(gpt.tok_emb.weight, self.params['wte'])
        
        for b in range(len(self.params["blocks"])):
            q_w, k_w, v_w = np.split((self.params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.weight = GPT2WeightsLoader.assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
            gpt.trf_blocks[b].att.W_key.weight = GPT2WeightsLoader.assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
            gpt.trf_blocks[b].att.W_value.weight = GPT2WeightsLoader.assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split(
                (self.params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.bias = GPT2WeightsLoader.assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
            gpt.trf_blocks[b].att.W_key.bias = GPT2WeightsLoader.assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
            gpt.trf_blocks[b].att.W_value.bias = GPT2WeightsLoader.assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

            gpt.trf_blocks[b].att.out_proj.weight = GPT2WeightsLoader.assign(gpt.trf_blocks[b].att.out_proj.weight, self.params["blocks"][b]["attn"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].att.out_proj.bias = GPT2WeightsLoader.assign(gpt.trf_blocks[b].att.out_proj.bias, self.params["blocks"][b]["attn"]["c_proj"]["b"])

            gpt.trf_blocks[b].ff.layers[0].weight = GPT2WeightsLoader.assign(gpt.trf_blocks[b].ff.layers[0].weight, self.params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            gpt.trf_blocks[b].ff.layers[0].bias = GPT2WeightsLoader.assign(gpt.trf_blocks[b].ff.layers[0].bias, self.params["blocks"][b]["mlp"]["c_fc"]["b"])
            gpt.trf_blocks[b].ff.layers[2].weight = GPT2WeightsLoader.assign(gpt.trf_blocks[b].ff.layers[2].weight, self.params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].ff.layers[2].bias = GPT2WeightsLoader.assign(gpt.trf_blocks[b].ff.layers[2].bias, self.params["blocks"][b]["mlp"]["c_proj"]["b"])

            gpt.trf_blocks[b].norm1.scale = GPT2WeightsLoader.assign(gpt.trf_blocks[b].norm1.scale, self.params["blocks"][b]["ln_1"]["g"])
            gpt.trf_blocks[b].norm1.shift = GPT2WeightsLoader.assign(gpt.trf_blocks[b].norm1.shift, self.params["blocks"][b]["ln_1"]["b"])
            gpt.trf_blocks[b].norm2.scale = GPT2WeightsLoader.assign(gpt.trf_blocks[b].norm2.scale, self.params["blocks"][b]["ln_2"]["g"])
            gpt.trf_blocks[b].norm2.shift = GPT2WeightsLoader.assign(gpt.trf_blocks[b].norm2.shift, self.params["blocks"][b]["ln_2"]["b"])

        gpt.final_norm.scale = GPT2WeightsLoader.assign(gpt.final_norm.scale, self.params["g"])
        gpt.final_norm.shift = GPT2WeightsLoader.assign(gpt.final_norm.shift, self.params["b"])
        gpt.out_head.weight = GPT2WeightsLoader.assign(gpt.out_head.weight, self.params["wte"])

    @staticmethod
    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(torch.tensor(right))

            

