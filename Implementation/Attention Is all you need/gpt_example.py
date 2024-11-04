import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import string

# =========================
# 1. Encoding Utilities
# =========================

def get_encoding_utilities():
    """
    Initializes and returns encoding utilities including vocabulary mappings and preprocessing functions.
    """
    # Define word-to-integer mappings for encoder and decoder
    word_to_int_encoder = {"the": 1, "cat": 2, "sat": 3, "on": 4, "mat": 5}
    int_to_word_decoder = {
        6: "it", 7: "sounds", 8: "like", 9: "you're",
        10: "quoting", 11: "a", 12: "classic", 13: "simple", 14: "sentence"
    }
    
    # Function to preprocess sentences: lowercase and remove punctuation
    def preprocess_sentence(sentence):
        sentence = sentence.lower()
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        return sentence
    
    # Function to convert sentence to numbers using encoder vocabulary
    def sentence_to_numbers_encoder(sentence):
        sentence = preprocess_sentence(sentence)
        return [word_to_int_encoder.get(word, 0) for word in sentence.split()]
    
    # Function to convert numbers to sentence using decoder vocabulary
    def numbers_to_sentence_decoder(numbers):
        # Exclude padding index '0' if present
        return " ".join([int_to_word_decoder.get(num, "") for num in numbers if num != 0])
    
    return sentence_to_numbers_encoder, numbers_to_sentence_decoder

# =========================
# 2. Embedding Layers
# =========================

def get_embedding_layer(vocab_size, embedding_dim=16):
    """
    Initializes and returns an embedding layer.
    """
    embedding_layer = nn.Embedding(vocab_size + 1, embedding_dim)  # +1 for padding (0)
    return embedding_layer

# =========================
# 3. Positional Encoding
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# =========================
# 4. Transformer Components
# =========================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        # Final linear layer
        output = self.out_linear(attn_output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=64, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.linear2(self.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, mask=None):
        # Self-Attention
        attn_output = self.self_attn(src, src, src, mask)
        src = self.norm1(src + self.dropout(attn_output))
        
        # Feed Forward
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        # Self-Attention
        attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn_output))
        
        # Encoder-Decoder Attention
        attn_output = self.enc_dec_attn(tgt, enc_output, enc_output, src_mask)
        tgt = self.norm2(tgt + self.dropout(attn_output))
        
        # Feed Forward
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_output))
        return tgt

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=16, num_heads=2, num_layers=2, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size + 1, d_model)  # +1 for padding (0)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size + 1, d_model)  # +1 for padding (0)
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size + 1)  # +1 for padding (0)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoder
        src_emb = self.encoder_embedding(src)
        src_emb = self.pos_encoder(src_emb)
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # Decoder
        tgt_emb = self.decoder_embedding(tgt)
        tgt_emb = self.pos_decoder(tgt_emb)
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # Output Layer
        output = self.fc_out(dec_output)
        
        # Calculate loss
        loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        
        return output, loss

# =========================
# 5. Preparing Input and Output
# =========================

def prepare_input_output(sentence_to_numbers, encoder_sentence, decoder_sentence):
    """
    Encodes the input and output sentences and converts them to tensors.
    """
    # Encode the sentences
    encoded_src = sentence_to_numbers(encoder_sentence)
    print("Encoded Source:", encoded_src)
    encoded_tgt = sentence_to_numbers(decoder_sentence)
    print("Encoded Target:", encoded_tgt)
    print()
    
    # Convert to tensors and add batch dimension
    src_tensor = torch.tensor(encoded_src).unsqueeze(0)  # Shape: (1, src_seq_length)
    tgt_tensor = torch.tensor(encoded_tgt).unsqueeze(0)  # Shape: (1, tgt_seq_length)
    
    return src_tensor, tgt_tensor


#  ===================
#  Create Transformer Model
#  ===================
def create_transformer_model(src_vocab_size, tgt_vocab_size, d_model=16, num_heads=2, num_layers=2, dropout=0.1):
    """
    Creates and returns a Transformer model with the specified parameters.
    """
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
    return model


# =========================
# 6. Main Function
# =========================

def main():
    # Initialize encoding utilities
    sentence_to_numbers, numbers_to_sentence = get_encoding_utilities()
    
    # Define vocab sizes based on the encoding utilities
    # Assuming the highest integer in word_to_int_encoder is the vocab size
    vocab_size_encoder = 5  # "the", "cat", "sat", "on", "mat"
    vocab_size_decoder = 14  # "it", "sounds", "like", "you're", "quoting", "a", "classic", "simple", "sentence"
    
    # Initialize embedding layers
    embedding_dim = 16
    encoder_embedding = get_embedding_layer(vocab_size_encoder, embedding_dim)
    decoder_embedding = get_embedding_layer(vocab_size_decoder, embedding_dim)
    
    # Initialize the Transformer model
    model = Transformer(
        src_vocab_size=vocab_size_encoder,
        tgt_vocab_size=vocab_size_decoder,
        d_model=embedding_dim,
        num_heads=2,
        num_layers=2,
        dropout=0.1
    )
    
    # Define input and output sentences
    encoder_sentence = "The cat sat on the mat."
    decoder_sentence = "It sounds like you're quoting a classic simple sentence!"
    
    # Prepare input and output tensors
    src_tensor, tgt_tensor = prepare_input_output(sentence_to_numbers, encoder_sentence, decoder_sentence)
    
    # Create a triangular mask for the target to prevent attention to future tokens
    seq_length = tgt_tensor.size(1)
    tril = torch.tril(torch.ones(seq_length, seq_length)).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_length, seq_length)
    tgt_mask = tril  # Using float mask
    
    # Forward pass through the model with the target mask
    output = model(src_tensor, tgt_tensor, tgt_mask=tgt_mask)
    
    print("Transformer Output Shape:", output.shape)  # Expected: (1, tgt_seq_length, tgt_vocab_size + 1)
    
    # Prevent the model from predicting '0' by setting its logit to -inf
    output[:, :, 0] = -float('inf')
    
    # Get predicted token IDs by taking the argmax
    predictions = torch.argmax(F.softmax(output, dim=-1), dim=-1)
    print("Predicted Token IDs:", predictions)
    
    # Convert predicted token IDs back to words
    predicted_sentence = numbers_to_sentence(predictions.squeeze().tolist())
    print("Predicted Sentence:", predicted_sentence)

if __name__ == "__main__":
    main()
