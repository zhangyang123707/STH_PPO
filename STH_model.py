import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert self.head_dim * n_heads == embed_dim, "embed_dim must be divisible by n_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并重塑为多头
        Q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        output = self.out_proj(context)
        return output, attn_weights

class SpatialAttention(nn.Module):
    """空间注意力：处理邻居路口间的关系"""
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super(SpatialAttention, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, neighbor_x, mask=None):
        # x: [batch, num_agents, embed_dim]
        # neighbor_x: [batch, num_agents, num_neighbors, embed_dim]
        batch_size, num_agents, num_neighbors, embed_dim = neighbor_x.shape
        
        # 将邻居特征展平用于注意力计算
        neighbor_x_flat = neighbor_x.view(batch_size, num_agents * num_neighbors, embed_dim)
        
        # 空间注意力
        attn_out, attn_weights = self.attention(x, neighbor_x_flat, neighbor_x_flat, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x, attn_weights

class TemporalAttention(nn.Module):
    """时间注意力：处理历史状态序列"""
    def __init__(self, embed_dim, n_heads, history_len, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.history_len = history_len
        
    def forward(self, x, history_x, mask=None):
        # x: [batch, num_agents, embed_dim]
        # history_x: [batch, num_agents, history_len, embed_dim]
        batch_size, num_agents, history_len, embed_dim = history_x.shape
        
        # 将历史特征展平用于注意力计算
        history_x_flat = history_x.view(batch_size, num_agents * history_len, embed_dim)
        
        # 时间注意力
        attn_out, attn_weights = self.attention(x, history_x_flat, history_x_flat, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x, attn_weights

class FeatureFusion(nn.Module):
    """特征融合模块：融合空间和时间特征"""
    def __init__(self, embed_dim, dropout=0.1):
        super(FeatureFusion, self).__init__()
        self.spatial_proj = nn.Linear(embed_dim, embed_dim)
        self.temporal_proj = nn.Linear(embed_dim, embed_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, spatial_feat, temporal_feat):
        # 特征投影
        spatial_proj = self.spatial_proj(spatial_feat)
        temporal_proj = self.temporal_proj(temporal_feat)
        
        # 门控融合
        gate_input = torch.cat([spatial_proj, temporal_proj], dim=-1)
        gate = self.fusion_gate(gate_input)
        
        # 加权融合
        fused_feat = gate * spatial_proj + (1 - gate) * temporal_proj
        output = self.output_proj(fused_feat)
        
        return self.dropout(output)

class STHANModel(nn.Module):
    """
    时空混合注意力网络（Spatio-Temporal Hybrid Attention Network）
    用于交通信号控制的特征提取与决策。
    """
    def __init__(self, obs_dim, action_dim, num_agents, history_len=4, embed_dim=128, n_heads=4, 
                 num_layers=2, dropout=0.1):
        super(STHANModel, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.history_len = history_len
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.num_layers = num_layers
        
        # 特征嵌入层
        self.obs_embedding = nn.Linear(obs_dim, embed_dim)
        self.neighbor_embedding = nn.Linear(obs_dim, embed_dim)
        self.history_embedding = nn.Linear(obs_dim, embed_dim)
        
        # 空间注意力层
        self.spatial_attention_layers = nn.ModuleList([
            SpatialAttention(embed_dim, n_heads, dropout) for _ in range(num_layers)
        ])
        
        # 时间注意力层
        self.temporal_attention_layers = nn.ModuleList([
            TemporalAttention(embed_dim, n_heads, history_len, dropout) for _ in range(num_layers)
        ])
        
        # 特征融合层
        self.feature_fusion = FeatureFusion(embed_dim, dropout)
        
        # 分层策略网络
        # 高层策略：选择信号阶段/策略
        self.high_level_policy = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 4)  # 4个高层策略选项
        )
        
        # 低层策略：具体动作选择
        self.low_level_policy = nn.Sequential(
            nn.Linear(embed_dim + 4, embed_dim // 2),  # +4是高层策略的one-hot编码
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, action_dim)
        )
        
        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, obs, neighbor_obs, history_obs):
        """
        obs: 当前路口观测 [batch, num_agents, obs_dim]
        neighbor_obs: 邻居路口观测 [batch, num_agents, num_neighbors, obs_dim]
        history_obs: 历史观测 [batch, num_agents, history_len, obs_dim]
        
        Returns:
            high_level_logits: 高层策略logits [batch, num_agents, 4]
            low_level_logits: 低层策略logits [batch, num_agents, action_dim]
            value: 状态价值 [batch, num_agents, 1]
            spatial_attn_weights: 空间注意力权重
            temporal_attn_weights: 时间注意力权重
        """
        batch_size = obs.size(0)
        
        # 特征嵌入
        obs_embed = self.obs_embedding(obs)  # [batch, num_agents, embed_dim]
        neighbor_embed = self.neighbor_embedding(neighbor_obs)  # [batch, num_agents, num_neighbors, embed_dim]
        history_embed = self.history_embedding(history_obs)  # [batch, num_agents, history_len, embed_dim]
        
        # 多层时空注意力
        spatial_feat = obs_embed
        temporal_feat = obs_embed
        spatial_attn_weights = []
        temporal_attn_weights = []
        
        for i in range(self.num_layers):
            # 空间注意力
            spatial_feat, spatial_attn = self.spatial_attention_layers[i](spatial_feat, neighbor_embed)
            spatial_attn_weights.append(spatial_attn)
            
            # 时间注意力
            temporal_feat, temporal_attn = self.temporal_attention_layers[i](temporal_feat, history_embed)
            temporal_attn_weights.append(temporal_attn)
        
        # 特征融合
        fused_feat = self.feature_fusion(spatial_feat, temporal_feat)
        
        # 高层策略
        high_level_logits = self.high_level_policy(fused_feat)
        
        # 低层策略（结合高层策略信息）
        high_level_probs = F.softmax(high_level_logits, dim=-1)
        low_level_input = torch.cat([fused_feat, high_level_probs], dim=-1)
        low_level_logits = self.low_level_policy(low_level_input)
        
        # 状态价值
        value = self.value_net(fused_feat)
        
        return high_level_logits, low_level_logits, value, spatial_attn_weights, temporal_attn_weights
    
    def get_action_probs(self, obs, neighbor_obs, history_obs):
        """获取动作概率分布"""
        high_level_logits, low_level_logits, value, _, _ = self.forward(obs, neighbor_obs, history_obs)
        
        high_level_probs = F.softmax(high_level_logits, dim=-1)
        low_level_probs = F.softmax(low_level_logits, dim=-1)
        
        return high_level_probs, low_level_probs, value
    
    def get_action_log_probs(self, obs, neighbor_obs, history_obs, high_level_actions, low_level_actions):
        """获取动作的对数概率"""
        high_level_logits, low_level_logits, value, _, _ = self.forward(obs, neighbor_obs, history_obs)
        
        high_level_log_probs = F.log_softmax(high_level_logits, dim=-1)
        low_level_log_probs = F.log_softmax(low_level_logits, dim=-1)
        
        # 收集对应动作的对数概率
        batch_size, num_agents = obs.size(0), obs.size(1)
        high_level_log_probs_selected = torch.gather(
            high_level_log_probs.view(-1, 4), 1, 
            high_level_actions.view(-1, 1)
        ).view(batch_size, num_agents)
        
        low_level_log_probs_selected = torch.gather(
            low_level_log_probs.view(-1, self.action_dim), 1,
            low_level_actions.view(-1, 1)
        ).view(batch_size, num_agents)
        
        return high_level_log_probs_selected, low_level_log_probs_selected, value 