import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        """
        Params:
            d_model: dimension of the embedded vector
            vocab_size: size of vocabulary
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # create embedding layer using built-in pytorch module.
        self.embedding = nn.Embedding(vocab_size, d_model)  # return the vector of size d_model for a given vocab_size

    def forward(self, x):
        # return embedded vector.
        # The term " * math.sqrt(self.d_model)" is added according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Params:
            d_model: embedding vector size
            seq_len: maximum sequence/sentence length input to the model
            dropout: dropout rate to prevent the model from overfitting
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)  # using built-in torch dropout function

        # Create a positional encoding matrix of shape (seq_len, d_model)
        self.pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1) for the positional encoding calculation
        # .un_squeeze(1) add a dimension at the position 1 from (seq_len) -> (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # the division term from the paper's equation (it's ok to not understand this one)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (math.log(10000.0) / d_model))

        # Apply the sin() to even position and cos() to the odd position
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        # Make the pe to support for batch (multi-sequence)
        self.pe = self.pe.unsqueeze(0)  # add a batch's dimension in the front -> (1, seq_len, d_model)

        self.register_buffer('pe', self.pe)     # make the positional encoding matrix un-learnable.

    def forward(self, x):
        # return embedded vector + positional encoding
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        # since we want to fix the positional encoding matrix
        # so using "requires_grad_(False)" the matrix will not update during learning process
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        # define params for the normalization: eps (epsilon), alpha, and bias
        # All these params are used in the normalization formular in forward()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))    # Multiplied, learnable parameter
        self.bias = nn.Parameter(torch.zeros(1))    # Added, learnable parameter

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)   # calculate mean
        std = x.std(dim=-1, keepdim=True)     # std
        return self.alpha * (x - mean) / (std + self.eps) + self.bias   # return normalized values

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        Params:
            d_model: input and output dimension
            d_ff: inner-layer dimension
        """
        super().__init__()
        # The original transformer has 2 linear layer
        self.linear_1 = nn.Linear(d_model, d_ff)    # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)    # W2 and b2

    def forward(self, x):
        # input sentence/sequence  = (Batch, seq_len, d_model)
        # (Batch, seq_len, d_model) --> linear_1 (Batch, seq_len, dff) --> linear_2 (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.attention_scores = None
        self.d_model = d_model
        self.h = h  # attention heads

        # check whether d_model can be divided by head
        assert d_model % h == 0, "d_model is not divisible by h"

        # weight metrics according to the paper
        self.d_k = d_model // h     # d_model/dimension for each head
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv

        self.w_o = nn.Linear(d_model, d_model)  # Wo : weight metrics for the output
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # query shape: (batch, h, seq_len, d_k)
        d_k = query.shape[-1]   # can use either query/key shape[-1]

        # calculate attention score using attention formular
        # the @ symbol means matrix multiplication
        # transpose key into shape: (batch, h, dk, seq_len) to be able to multiple with query
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)     # (batch, h, seq_len, seq_len)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)      # fill mask with -1e9

        # apply softmax
        attention_scores = attention_scores.softmax(dim = -1)   # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len) * (batch, h, seq_len, d_k) = (batch, h, seq_len, d_k)
        return  (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        # create query, key, value metrics by multiply each vector with their weight metrics
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)   # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # divide each metric into different heads using .view() from pytorch
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> transpose to (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # calculate attention score
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # concat each head together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # contiguous() means tensor x is in a contiguous memory layout, which is necessary for certain operations.
        # -1 in .view() means it will be automatically calculated based on the other dimensions.

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return  self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # the residual also has the normalization
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # the equation is slightly different from the paper, but most people implement like this.
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # create 2 residual connection using ModuleList
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # src_mask is used to mask the padding tokens to prevent the encoder to learn from them

        # attention + residual connection
        # here the q,k,v is all the same which is x
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))

        # feedforward + residual
        x = self.residual_connection[1](x, self.feed_forward_block)

        return x

class Encoder(nn.Module):
    # class contains stacks/layers of encoder blocks
    def __init__(self, layers: nn.ModuleList ) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        # iterate through each encoder layer
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    # decoder block is similar to encoder block but the q,k,v is different
    # the decoder block contains 2 attention blocks: self-attention and cross-attention (key, value from encoder)

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # self attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # cross attention block
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return  self.norm(x)

class ProjectionLayer(nn.Module):
    # Last linear layer in the transformer architecture. It is used to project the output back to the vocabulary
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)    # used log_softmax for mathematical stability

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    # create 3 methods (encode, decode, project) for flexible used instead of the forward()
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

# Method for building the Transformer model
def build_transformer(scr_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                      d_model: int = 512, N: int = 6, h: int = 8, d_ff: int= 2048, dropout: float = 0.1) -> Transformer:
    # create the embedding layers
    src_embed = InputEmbedding(d_model, scr_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # create the positional embedding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # create the decoder block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)   # initialize the parameters using Xavier initialization

    return transformer