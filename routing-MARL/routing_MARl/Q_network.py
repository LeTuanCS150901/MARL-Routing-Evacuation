import torch
import torch.nn as nn
import torch.nn.functional as F

from read_config import get_config
from logs.logger import logger  # import our custom logger

class QNetwork(nn.Module):  
    def __init__(self, obs_dim, action_dim, dropout_p=0.1):  
        super().__init__()  

        self.CONFIG = get_config()  

        # Transformer encoder params
        self.num_heads = self.CONFIG['model']['encoder']['num_heads']
        self.head_dim = self.CONFIG['model']['encoder']['head_dim']
        self.hidden_dim = self.num_heads * self.head_dim  
        self.num_encoder_layers = self.CONFIG['model']['encoder']['num_encoder_layers']
        
        # Q network layers
        self.hidden_sizes = self.CONFIG['model']['q_network']['hidden_sizes']

        # memory (optional): the last K entries of the observation are the car's last K
        # road indices (oldest first, padded with action_dim); an LSTM over their embeddings runs next to the encoder
        mem = self.CONFIG.get('memory') or {}
        self.mem_k = int(mem.get('window', 8)) if mem.get('enabled') else 0
        self.base_dim = obs_dim - self.mem_k
        self.action_dim = action_dim
        if self.mem_k:
            self.mem_hidden = int(mem.get('hidden', 64))
            self.road_emb = nn.Embedding(action_dim + 1, int(mem.get('emb_dim', 32)), padding_idx=action_dim)
            self.lstm = nn.LSTM(int(mem.get('emb_dim', 32)) + 1, self.mem_hidden, batch_first=True)
            self.mem_norm = nn.LayerNorm(self.mem_hidden)

        # --- Input projection ---
        self.input_proj = nn.Linear(self.base_dim, self.hidden_dim)
    
        try:  
            self.feature_dim_disaster = self.CONFIG['disaster']['fire']['feature_dim']  
        except Exception as e:  
            logger.warning("No disaster feature dimension found in config, set to 1")  
            self.feature_dim_disaster = 1  
        # --- Disaster projection ---  
        self.disaster_encoder = nn.Linear(self.feature_dim_disaster, self.hidden_dim)  

        # --- Transformer encoder for inter-vehicle attention ---
        # model.encoder.enabled: false = no-encoder ablation: no transformer and no
        # group mean; each car's Q comes from its own projected observation and hazard input only. Default true = unchanged.
        self.use_encoder = bool(self.CONFIG['model']['encoder'].get('enabled', True))
        self.norm_first = bool(self.CONFIG['model']['encoder'].get('norm_first', False))
        # model.encoder.type: 'transformer' (default = every earlier model) or
        # 'lstm' = the paper's Coop-Eva-LSTM encoder (branch marl_coope_lstm_randers_gpu_v1): unidirectional nn.LSTM over the
        # cars of the group, hidden = num_heads * head_dim, num_encoder_layers layers, dropout between layers
        self.encoder_type = str(self.CONFIG['model']['encoder'].get('type', 'transformer')).lower()
        if self.encoder_type not in ('transformer', 'lstm'):
            raise ValueError(f"model.encoder.type must be 'transformer' or 'lstm', got {self.encoder_type!r}")
        if self.encoder_type == 'lstm' and not self.use_encoder:
            raise ValueError("model.encoder.type 'lstm' needs model.encoder.enabled: true")
        if self.use_encoder and self.encoder_type == 'lstm':
            self.lstm_encoder = nn.LSTM(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_encoder_layers,
                batch_first=True,
                dropout=dropout_p if self.num_encoder_layers > 1 else 0.0,
                bidirectional=False
            )
        elif self.use_encoder:
            encoder_layer = nn.TransformerEncoderLayer(  
                norm_first=self.norm_first,
                d_model=self.hidden_dim,  
                nhead=self.num_heads,  
                dim_feedforward=self.hidden_dim * 4,  
                dropout=dropout_p,  
                batch_first=True  
            )  
            # Pre-LN (model.encoder.norm_first: true) needs a final LayerNorm on the encoder output
            final_norm = nn.LayerNorm(self.hidden_dim) if self.norm_first else None
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers, norm=final_norm)  

        # --- Feedforward layers per vehicle ---
        n_parts = 3 if self.use_encoder else 2  # encoded, group_embed, disaster_embed | ablation: obs_proj, disaster_embed
        layer_sizes = [self.hidden_dim * n_parts + (self.mem_hidden if self.mem_k else 0)] + self.hidden_sizes  # (+ memory)
        self.hidden_layers = nn.ModuleList([  
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)  
        ])  
        self.bn_layers = nn.ModuleList([  
            nn.LayerNorm(layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)  
        ])  
        self.drop_layers = nn.ModuleList([  
            nn.Dropout(p=dropout_p) if 0 < i < len(layer_sizes) - 2 else nn.Identity()  
            for i in range(len(layer_sizes) - 1)  
        ])  

        # --- Output layer ---  
        self.output_layer = nn.Linear(layer_sizes[-1], action_dim)

    def forward(self, obs, disaster=None, padding_mask=None):  
        """  
        Forward pass for QNetwork.  

        - obs: [B, max_N, obs_dim] (batched groups, padded)  
          OR [N, obs_dim] (single group, inference) -> auto-unsqueeze to [1, N, obs_dim]  
          OR [1, obs_dim] (single vehicle) -> auto-unsqueeze to [1, 1, obs_dim]  
        - disaster: Same shape as obs, but [B, max_N, dis_dim] (or None)  
        - padding_mask: [B, max_N] (True for padded positions, optional for inference)  
        
        Returns: [N_total, action_dim] (flattened across batches for easy gathering)  
        """  
        # Handle inference cases (single group or vehicle)  
        if obs.dim() == 2:  # [N, obs_dim] -> [1, N, obs_dim]  
            obs = obs.unsqueeze(0)  
            if disaster is not None:  
                disaster = disaster.unsqueeze(0)  
            if padding_mask is not None:  
                padding_mask = padding_mask.unsqueeze(0)  
        elif obs.dim() == 1:  # [obs_dim] -> [1, 1, obs_dim]  
            obs = obs.unsqueeze(0).unsqueeze(0)  
            if disaster is not None:  
                disaster = disaster.unsqueeze(0).unsqueeze(0)  
            if padding_mask is not None:  
                padding_mask = padding_mask.unsqueeze(0).unsqueeze(0)  

        B, max_N, obs_dim = obs.shape
        device = self.input_proj.weight.device
        obs = obs.to(device)
        if self.mem_k:
            if obs_dim != self.base_dim + self.mem_k:
                raise ValueError(f"memory on: observation has {obs_dim} entries, expected {self.base_dim} + {self.mem_k} road indices")
            hist = obs[..., self.base_dim:].round().long().clamp(0, self.action_dim)        # [B, max_N, K]
            obs = obs[..., :self.base_dim]
        if disaster is not None:  
            disaster = disaster.to(device)  
        if padding_mask is not None:  
            padding_mask = padding_mask.to(device)  

        # Project obs: [B, max_N, hidden_dim]  
        obs_proj = self.input_proj(obs)  

        if self.use_encoder:
            # --- Transformer encoding with padding mask ---  
            if self.encoder_type == 'lstm':
                if padding_mask is not None:   # as the paper LSTM: pack the real cars of each group, pad back to max_N
                    lengths = (~padding_mask).sum(dim=1).cpu().clamp(min=1)
                    packed = nn.utils.rnn.pack_padded_sequence(obs_proj, lengths, batch_first=True, enforce_sorted=False)
                    out_packed, _ = self.lstm_encoder(packed)
                    x_encoded, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=max_N)
                else:
                    x_encoded, _ = self.lstm_encoder(obs_proj)  # [B, max_N, hidden_dim]
            else:
                x_encoded = self.transformer(obs_proj, src_key_padding_mask=padding_mask)  # [B, max_N, hidden_dim]  

            # --- Group embedding (mean over valid vehicles, ignoring padding) ---  
            if padding_mask is not None:  
                # Mask padded positions by setting to 0 before mean  
                masked_encoded = x_encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)  
                group_embed = masked_encoded.sum(dim=1) / (max_N - padding_mask.sum(dim=1, keepdim=True) + 1e-8)  # [B, hidden_dim]  
            else:  
                group_embed = x_encoded.mean(dim=1)  # [B, hidden_dim]  
            group_embed = group_embed.unsqueeze(1).repeat(1, max_N, 1)  # [B, max_N, hidden_dim]  

        # --- Disaster embedding ---  
        if disaster is None:  
            disaster_embed = torch.full((B, max_N, self.hidden_dim), -1.0, device=device)  
        else:  
            disaster_proj = self.disaster_encoder(disaster)  # [B, max_N, hidden_dim]  
            # Assuming mean over disaster features if multi-dimensional; adjust if needed  
            disaster_embed = disaster_proj  # Or .mean(dim=2) if disaster is [B, max_N, dis_dim] >1  

        # --- Concatenate x_encoded + group_embed + disaster_embed ---  
        if self.use_encoder:
            concat_input = torch.cat([x_encoded, group_embed, disaster_embed], dim=2)  # [B, max_N, 3*hidden_dim]
        else:
            concat_input = torch.cat([obs_proj, disaster_embed], dim=2)  # ablation: [B, max_N, 2*hidden_dim], no car sees another
        if self.mem_k:
            emb = self.road_emb(hist.view(B * max_N, self.mem_k))                              # [B*max_N, K, emb]
            now = torch.zeros(B * max_N, self.mem_k, 1, device=device)
            now[:, -1] = 1.0                                                                   # current-road flag
            _, (h_n, _) = self.lstm(torch.cat([emb, now], dim=2))
            memory = self.mem_norm(h_n[-1]).view(B, max_N, self.mem_hidden)                    # [B, max_N, H]
            concat_input = torch.cat([concat_input, memory], dim=2)
        # Optional: true residual connection  
        # concat_input = concat_input + obs_proj.repeat(1, 1, 3)  # Match dims if needed  

        # --- MLP (process per vehicle) ---  
        h = concat_input.view(B * max_N, -1)  # Flatten to [B*max_N, 3*hidden_dim] for efficient MLP  
        for linear, norm, drop in zip(self.hidden_layers, self.bn_layers, self.drop_layers):  
            # change the order here to LayerNorm -> ReLU -> Dropout

            h = norm(torch.relu(linear(h)))
            # h = drop(torch.relu(norm(linear(h))))  
        q_values = self.output_layer(h)  # [B*max_N, action_dim]  

        # Return flattened [N_total, action_dim] (including padded, but you'll ignore them via gathering)  
        return q_values