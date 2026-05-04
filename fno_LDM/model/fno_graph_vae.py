import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class FeedForward(nn.Module):
    """A simple Feed Forward network block for Transformer-style architectures."""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# --- VAE specific to FNO structure ---
class FNO_GraphVAE(nn.Module):
    def __init__(self, fno_model, internal_dim: int, latent_dim: int,
                 num_heads: int, num_attn_layers=1,
                 kl_weight=1.0, dropout=0.1, layer_recons_loss=False):
        super().__init__()
        assert internal_dim % num_heads == 0, f"Internal dimension ({internal_dim}) must be divisible by num_heads ({num_heads})"

        self.fno_model = fno_model

        self.internal_dim = internal_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_attn_layers = num_attn_layers
        
        self.layer_recons_loss = layer_recons_loss
        self.kl_weight = kl_weight

        # --- Parse FNO structure and create projection layers ---
        self.fno_structure = OrderedDict() # Store layer info (type, name, N, D_in)
        self.input_projs_dict = nn.ModuleDict()
        self.output_inv_projs_dict = nn.ModuleDict()
        self.node_types = ['lift', 'block', 'proj']
        self._node_counts_per_layer_list = [] # To store N per layer for splitting

        state_dict = fno_model.state_dict()
        layer_counter = 0

        # 1. Lifting Layers
        self.fno_structure['lift'] = []
        lift_proj_list = nn.ModuleList()
        lift_inv_proj_list = nn.ModuleList()
        for i, layer in enumerate(fno_model.lifting.fcs):
            weight_key = f'lifting.fcs.{i}.weight'
            bias_key = f'lifting.fcs.{i}.bias'
            weight = state_dict[weight_key] # D_out, D_in, 1 or D_out, D_in
            D_out, D_in = weight.shape[0], weight.shape[1]
            bias_exists = bias_key in state_dict
            D_layer = (D_in + 1) if bias_exists else D_in # Actual input feature dim

            self.fno_structure['lift'].append({'name': f'lift_{i}', 'N': D_out, 'D_in': D_layer})
            self._node_counts_per_layer_list.append(D_out)
            lift_proj_list.append(nn.Linear(D_layer, internal_dim))
            lift_inv_proj_list.append(nn.Linear(internal_dim, D_layer))
            layer_counter += 1
        self.input_projs_dict['lift'] = lift_proj_list
        self.output_inv_projs_dict['lift'] = lift_inv_proj_list

        # 2. FNO Block Layers
        self.fno_structure['block'] = []
        block_proj_list = nn.ModuleList()
        block_inv_proj_list = nn.ModuleList()
        n_fno_layers = len(fno_model.fno_blocks.convs)
        for i in range(n_fno_layers):
            w1_key = f'fno_blocks.convs.{i}.weights1'
            w2_key = f'fno_blocks.convs.{i}.weights2'
            skip_key = f'fno_blocks.fno_skips.{i}.conv.weight'
            w1 = state_dict[w1_key] # C_out, C_in, mode_h, mode_w
            w2 = state_dict[w2_key]
            skip = state_dict[skip_key] # C_out, C_in, 1
            C_out, C_in = w1.shape[0], w1.shape[1]
            # Calculate actual input feature dim D_layer based on concatenation
            D_layer = w1.numel() // C_out * 2 + w2.numel() // C_out * 2 + skip.numel() // C_out

            self.fno_structure['block'].append({'name': f'block_{i}', 'N': C_out, 'D_in': D_layer})
            self._node_counts_per_layer_list.append(C_out)
            block_proj_list.append(nn.Linear(D_layer, internal_dim))
            block_inv_proj_list.append(nn.Linear(internal_dim, D_layer))
            layer_counter += 1
        self.input_projs_dict['block'] = block_proj_list
        self.output_inv_projs_dict['block'] = block_inv_proj_list

        # 3. Projection Layers
        self.fno_structure['proj'] = []
        proj_proj_list = nn.ModuleList()
        proj_inv_proj_list = nn.ModuleList()
        for i, layer in enumerate(fno_model.projection.fcs):
            weight_key = f'projection.fcs.{i}.weight'
            bias_key = f'projection.fcs.{i}.bias'
            weight = state_dict[weight_key] # D_out, D_in, 1 or D_out, D_in
            D_out, D_in = weight.shape[0], weight.shape[1]
            bias_exists = bias_key in state_dict
            D_layer = (D_in + 1) if bias_exists else D_in # Actual input feature dim

            self.fno_structure['proj'].append({'name': f'proj_{i}', 'N': D_out, 'D_in': D_layer})
            self._node_counts_per_layer_list.append(D_out)
            proj_proj_list.append(nn.Linear(D_layer, internal_dim))
            proj_inv_proj_list.append(nn.Linear(internal_dim, D_layer))
            layer_counter += 1
        self.input_projs_dict['proj'] = proj_proj_list
        self.output_inv_projs_dict['proj'] = proj_inv_proj_list

        # --- VAE Components ---
        self.encoder_proj_mu_logvar = nn.Sequential(
            nn.Linear(internal_dim, internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, latent_dim * 2)
        )
        self.decoder_latent_proj = nn.Sequential(
            nn.Linear(latent_dim, internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, internal_dim)
        )

        # --- Encoder Attention Blocks ---
        self.encoder_attn_layers = nn.ModuleList([])
        self.encoder_ffn_layers = nn.ModuleList([])
        self.encoder_norm1_layers = nn.ModuleList([])
        self.encoder_norm2_layers = nn.ModuleList([])
        for _ in range(num_attn_layers):
            self.encoder_norm1_layers.append(nn.LayerNorm(internal_dim))
            self.encoder_attn_layers.append(
                nn.MultiheadAttention(embed_dim=internal_dim, num_heads=num_heads,
                                        dropout=dropout, batch_first=True)
            )
            self.encoder_norm2_layers.append(nn.LayerNorm(internal_dim))
            # FFN hidden dim often 4*internal_dim in Transformers
            self.encoder_ffn_layers.append(FeedForward(internal_dim, internal_dim * 4, dropout))


        # --- Decoder Attention Blocks ---
        self.decoder_attn_layers = nn.ModuleList([])
        self.decoder_ffn_layers = nn.ModuleList([])
        self.decoder_norm1_layers = nn.ModuleList([])
        self.decoder_norm2_layers = nn.ModuleList([])
        for _ in range(num_attn_layers):
            self.decoder_norm1_layers.append(nn.LayerNorm(internal_dim))
            self.decoder_attn_layers.append(
                nn.MultiheadAttention(embed_dim=internal_dim, num_heads=num_heads,
                                        dropout=dropout, batch_first=True)
            )
            self.decoder_norm2_layers.append(nn.LayerNorm(internal_dim))
            # FFN hidden dim often 4*internal_dim in Transformers
            self.decoder_ffn_layers.append(FeedForward(internal_dim, internal_dim * 4, dropout))

        self.reset_parameters()

    def reset_parameters(self, mean=0.0, std=0.001): # Adjusted std based on typical transformer init
        """ Reset (initialize) all learnable parameters """
        def init_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=mean, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        def init_layernorm(m):
             if isinstance(m, nn.LayerNorm):
                  nn.init.constant_(m.bias, 0)
                  nn.init.constant_(m.weight, 1.0)

        # input proj / output proj
        for node_type in self.node_types:
            for proj_layer_list in self.input_projs_dict[node_type]:
                proj_layer_list.apply(init_linear)
            for inv_proj_layer_list in self.output_inv_projs_dict[node_type]:
                inv_proj_layer_list.apply(init_linear)

        # VAE specific projections
        init_linear(self.encoder_proj_mu_logvar)
        init_linear(self.decoder_latent_proj)
        
        # Encoder / Decoder Attention Block Layers
        for i in range(self.num_attn_layers):
            # Attention weights (handled by nn.MultiheadAttention internal init)
            # FFN layers
            self.encoder_ffn_layers[i].apply(init_linear)
            self.decoder_ffn_layers[i].apply(init_linear)
            # LayerNorm
            init_layernorm(self.encoder_norm1_layers[i])
            init_layernorm(self.encoder_norm2_layers[i])
            init_layernorm(self.decoder_norm1_layers[i])
            init_layernorm(self.decoder_norm2_layers[i])
        

    def encode(self, X_nodes_dict: dict):
        projected_nodes = []

        # Process and project layer by layer, preserving order
        for node_type in self.node_types:
            layer_list = X_nodes_dict.get(node_type, []) # Use .get for safety
            proj_layer_list = self.input_projs_dict[node_type]
            assert len(layer_list) == len(proj_layer_list), f"Mismatch in layer count for type {node_type}: {len(layer_list)} != {len(proj_layer_list)}"

            for i, X_layer in enumerate(layer_list):
                # Apply layer-specific projection
                X_proj = self.input_projs_dict[node_type][i](X_layer) # (B, N_layer, internal_dim)
                projected_nodes.append(F.gelu(X_proj))

        X_unified = torch.cat(projected_nodes, dim=1) # (B, N_total, internal_dim)

        # --- Attention Encoding part ---
        X_node_enc = X_unified
        for i in range(self.num_attn_layers):
            # --- Attention Block ---
            # 1. Layer Norm 1 -> Multi-Head Self-Attention -> Residual 1
            x_norm1 = self.encoder_norm1_layers[i](X_node_enc)
            attn_output, _ = self.encoder_attn_layers[i](x_norm1, x_norm1, x_norm1)
            X_node_enc = X_node_enc + attn_output # Residual connection 1

            # 2. Layer Norm 2 -> Feed Forward -> Residual 2
            x_norm2 = self.encoder_norm2_layers[i](X_node_enc)
            ffn_output = self.encoder_ffn_layers[i](x_norm2)
            X_node_enc = X_node_enc + ffn_output # Residual connection 2
            # --- End Attention Block ---

        mu_logvar = self.encoder_proj_mu_logvar(X_node_enc) # (B, N_total, latent_dim * 2)
        mu, log_var = mu_logvar.chunk(2, dim=-1) # (B, N_total, latent_dim) each

        return mu, log_var

    def reparameterize(self, mu, log_var):
        """ Standard VAE reparameterization trick. """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        B, N_total, H_dim = z.shape
        node_counts_per_layer = self._node_counts_per_layer_list

        # Check consistency
        if sum(node_counts_per_layer) != N_total:
             raise ValueError(f"Sum of node_counts {sum(node_counts_per_layer)} != N_total {N_total} in latent z.")

        # Project latent features back to internal dimension space
        X_node_dec_init = self.decoder_latent_proj(z) # (B, N_total, internal_dim)

        # --- Attention Decoding part ---
        X_node_dec = X_node_dec_init
        for i in range(self.num_attn_layers):
             # --- Attention Block ---
             # 1. Layer Norm 1 -> Multi-Head Self-Attention -> Residual 1
            x_norm1 = self.decoder_norm1_layers[i](X_node_dec)
            attn_output, _ = self.decoder_attn_layers[i](x_norm1, x_norm1, x_norm1)
            X_node_dec = X_node_dec + attn_output # Residual 1

            # 2. Layer Norm 2 -> Feed Forward -> Residual 2
            x_norm2 = self.decoder_norm2_layers[i](X_node_dec)
            ffn_output = self.decoder_ffn_layers[i](x_norm2)
            X_node_dec = X_node_dec + ffn_output # Residual 2
            # --- End Attention Block ---

        # --- Split and Inverse Project ---
        split_tensors_internal_D = torch.split(X_node_dec, node_counts_per_layer, dim=1)

        reconstructions_dict = {nt: [] for nt in self.node_types}
        tensor_idx = 0

        current_type_idx = 0
        layers_processed_in_current_type = 0
        total_layers_in_current_type = len(self.output_inv_projs_dict[self.node_types[current_type_idx]])

        for split_tensor in split_tensors_internal_D:
            current_node_type = self.node_types[current_type_idx]
            inv_proj_layer = self.output_inv_projs_dict[current_node_type][layers_processed_in_current_type]

            # Apply the correct inverse projection
            recon_tensor = inv_proj_layer(split_tensor) # (B, N_layer, D_orig_layer)
            reconstructions_dict[current_node_type].append(recon_tensor)

            # Move to the next layer/type
            layers_processed_in_current_type += 1
            if layers_processed_in_current_type == total_layers_in_current_type:
                current_type_idx += 1
                if current_type_idx < len(self.node_types):
                    layers_processed_in_current_type = 0
                    total_layers_in_current_type = len(self.output_inv_projs_dict[self.node_types[current_type_idx]])
                elif current_type_idx == len(self.node_types) and tensor_idx != len(split_tensors_internal_D) -1:
                    print("Warning: More split tensors than expected layers in fno_structure.")

            tensor_idx += 1 # Move to the next split tensor

        return reconstructions_dict

    def forward(self, X_nodes_dict: dict):
        """ Full VAE forward pass using standard MHA. """
        mu, log_var = self.encode(X_nodes_dict)
        z = self.reparameterize(mu, log_var)
        reconstructions_dict = self.decode(z)
        return reconstructions_dict, mu, log_var, z

    def training_losses(self, X_nodes_dict):
        """ Calculates VAE loss using standard MHA encoder/decoder. """
        X_nodes_dict_orig = X_nodes_dict # Keep original for comparison
        # 0. Forward pass
        reconstructions_dict, mu, log_var, z = self(X_nodes_dict) # Call updated forward

        # 1. Reconstruction Loss (Logic remains the same)
        type_recon_loss, type_num = 0., 0
        for node_type in self.node_types:
            orig_list = X_nodes_dict_orig.get(node_type)
            recon_list = reconstructions_dict.get(node_type)

            for i in range(len(orig_list)):
                X_orig = orig_list[i]
                X_recon = recon_list[i]
                
                if 'lift' in node_type and self.layer_recons_loss:
                    bias_dim = 1
                    loss_layer = F.mse_loss(X_recon[...,:-bias_dim], X_orig[...,:-bias_dim], reduction='mean') + \
                        F.mse_loss(X_recon[...,-bias_dim:], X_orig[...,-bias_dim:], reduction='mean')
                    type_num +=2
                elif 'block' in node_type and self.layer_recons_loss:
                    bias_dim = X_orig.shape[1]
                    conv_dim = (X_orig.shape[2] - bias_dim) // 2
                    assert conv_dim*2+bias_dim == X_orig.shape[2], f"Unexpected dimensions in block layer: {X_orig.shape}"
                    loss_layer = F.mse_loss(X_recon[...,:conv_dim], X_orig[...,:conv_dim], reduction='mean') + \
                        F.mse_loss(X_recon[...,conv_dim:2*conv_dim], X_orig[...,conv_dim:2*conv_dim], reduction='mean') + \
                        F.mse_loss(X_recon[...,2*conv_dim:], X_orig[...,2*conv_dim:], reduction='mean')
                    type_num +=3
                elif 'proj' in node_type and i < len(orig_list) - 1 and self.layer_recons_loss: # Last proj layer is not biased
                    bias_dim = 1
                    loss_layer = F.mse_loss(X_recon[...,:-bias_dim], X_orig[...,:-bias_dim], reduction='mean') + \
                    F.mse_loss(X_recon[...,-bias_dim:], X_orig[...,-bias_dim:], reduction='mean')
                    type_num +=2
                else:
                    loss_layer = F.mse_loss(X_recon, X_orig, reduction='mean')
                    type_num +=1
                
                type_recon_loss += loss_layer

        weighted_recon_loss = type_recon_loss / type_num

        # 2. KL Divergence Loss (Logic remains the same, applied per node latent)
        kld_element = 1 + log_var - mu.pow(2) - log_var.exp()
        kl_div_loss = -0.5 * torch.mean(kld_element)
        
        # 3. Total VAE Loss
        total_loss = weighted_recon_loss + self.kl_weight * kl_div_loss

        return total_loss, weighted_recon_loss, kl_div_loss
