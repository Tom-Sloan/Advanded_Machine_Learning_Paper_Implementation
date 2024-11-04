import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import string

# =========================
# 1. Encoding Utilities
# =========================

def encoding_utilities():
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

def embedding_layers(vocab_size_encoder, vocab_size_decoder, embedding_dim=16):
    """
    Initializes and returns encoder and decoder embedding layers.
    """
    embedding_encoder = nn.Embedding(vocab_size_encoder + 1, embedding_dim)  # +1 for padding (0)
    embedding_decoder = nn.Embedding(vocab_size_decoder + 1, embedding_dim)  # +1 for padding (0)
    return embedding_encoder, embedding_decoder

# =========================
# 3. Positional Encoding
# =========================

def positional_encoding(embedding_dim, max_len=10):
    """
    Creates positional encoding matrix.
    """
    pe = torch.zeros(1, max_len, embedding_dim)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe  # Shape: (1, max_len, embedding_dim)

# =========================
# 4. Transformer Components
# =========================

def transformer_components(embedding_dim, num_heads=2):
    """
    Initializes and returns transformer components such as linear layers and layer norms.
    """
    # Initialize linear layers for Query, Key, Value
    Q_linear = nn.Linear(embedding_dim, embedding_dim)
    K_linear = nn.Linear(embedding_dim, embedding_dim)
    V_linear = nn.Linear(embedding_dim, embedding_dim)
    
    # Initialize linear layer for output of multi-head attention
    out_linear = nn.Linear(embedding_dim, embedding_dim)
    
    # Initialize feed-forward network layers
    ff_linear1 = nn.Linear(embedding_dim, 64)
    ff_relu = nn.ReLU()
    ff_linear2 = nn.Linear(64, embedding_dim)
    ff_dropout = nn.Dropout(0.1)
    
    # Initialize layer normalization layers
    layer_norm1 = nn.LayerNorm(embedding_dim)
    layer_norm2 = nn.LayerNorm(embedding_dim)
    
    # Initialize dropout layer
    dropout = nn.Dropout(0.1)
    
    return {
        "Q_linear": Q_linear,
        "K_linear": K_linear,
        "V_linear": V_linear,
        "out_linear": out_linear,
        "ff_linear1": ff_linear1,
        "ff_relu": ff_relu,
        "ff_linear2": ff_linear2,
        "ff_dropout": ff_dropout,
        "layer_norm1": layer_norm1,
        "layer_norm2": layer_norm2,
        "dropout": dropout
    }

# =========================
# 5. Preparing Input and Output
# =========================

def prepare_input_output(sentence_to_numbers_encoder, encoder_sentence, decoder_sentence):
    """
    Encodes the input and output sentences and converts them to tensors.
    """
    # Encode the sentences
    encoded_src = sentence_to_numbers_encoder(encoder_sentence)
    print("Encoded Source:", encoded_src)
    encoded_tgt = sentence_to_numbers_encoder(decoder_sentence)
    print("Encoded Target:", encoded_tgt)
    print()
    
    # Convert to tensors and add batch dimension
    src_tensor = torch.tensor(encoded_src).unsqueeze(0)  # Shape: (1, src_seq_length)
    tgt_tensor = torch.tensor(encoded_tgt).unsqueeze(0)  # Shape: (1, tgt_seq_length)
    
    return src_tensor, tgt_tensor

# =========================
# 6. Forward Pass
# =========================

def forward_pass(src_tensor, tgt_tensor, embedding_encoder, embedding_decoder, pe, components, numbers_to_sentence_decoder):
    """
    Performs the forward pass through the transformer and returns predicted sentence.
    """
    # Transformer components
    Q_linear = components["Q_linear"]
    K_linear = components["K_linear"]
    V_linear = components["V_linear"]
    out_linear = components["out_linear"]
    ff_linear1 = components["ff_linear1"]
    ff_relu = components["ff_relu"]
    ff_linear2 = components["ff_linear2"]
    ff_dropout = components["ff_dropout"]
    layer_norm1 = components["layer_norm1"]
    layer_norm2 = components["layer_norm2"]
    dropout = components["dropout"]
    
    # =========================
    # Encoder
    # =========================
    
    # Step 1: Embed the source sentence
    src_embeddings = embedding_encoder(src_tensor)  # Shape: (1, src_seq_length, embedding_dim)
    
    # Step 2: Add positional encoding
    src_embeddings = src_embeddings + pe[:, :src_embeddings.size(1), :]  # Shape: (1, src_seq_length, embedding_dim)
    
    # Step 3: Compute Q, K, V matrices
    Q = Q_linear(src_embeddings)  # Shape: (1, src_seq_length, embedding_dim)
    K = K_linear(src_embeddings)  # Shape: (1, src_seq_length, embedding_dim)
    V = V_linear(src_embeddings)  # Shape: (1, src_seq_length, embedding_dim)
    
    # Step 4: Split Q, K, V for multi-head attention
    num_heads = 2
    head_dim = Q.size(-1) // num_heads
    def split_heads(x):
        batch_size, seq_length, dim = x.size()
        x = x.view(batch_size, seq_length, num_heads, head_dim)
        return x.transpose(1,2)  # Shape: (batch_size, num_heads, seq_length, head_dim)
    
    Q = split_heads(Q)
    K = split_heads(K)
    V = split_heads(V)
    
    # Step 5: Scaled Dot-Product Attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)  # Shape: (1, num_heads, src_seq_length, src_seq_length)
    attn_weights = F.softmax(scores, dim=-1)  # Shape: (1, num_heads, src_seq_length, src_seq_length)
    attn_output = torch.matmul(attn_weights, V)  # Shape: (1, num_heads, src_seq_length, head_dim)
    
    # Step 6: Concatenate heads
    attn_output = attn_output.transpose(1,2).contiguous().view(1, src_tensor.size(1), -1)  # Shape: (1, src_seq_length, embedding_dim)
    
    # Step 7: Apply output linear layer
    attn_output = out_linear(attn_output)  # Shape: (1, src_seq_length, embedding_dim)
    
    # Step 8: Add residual connection and apply layer normalization
    src_embeddings = layer_norm1(src_embeddings + attn_output)  # Shape: (1, src_seq_length, embedding_dim)
    
    # =========================
    # Decoder
    # =========================
    
    # Step 1: Embed the target sentence
    tgt_embeddings = embedding_decoder(tgt_tensor)  # Shape: (1, tgt_seq_length, embedding_dim)
    
    # Step 2: Add positional encoding
    tgt_embeddings = tgt_embeddings + pe[:, :tgt_embeddings.size(1), :]  # Shape: (1, tgt_seq_length, embedding_dim)
    
    # Step 3: Compute Q, K, V matrices for decoder self-attention
    Q_dec = Q_linear(tgt_embeddings)  # Shape: (1, tgt_seq_length, embedding_dim)
    K_dec = K_linear(tgt_embeddings)  # Shape: (1, tgt_seq_length, embedding_dim)
    V_dec = V_linear(tgt_embeddings)  # Shape: (1, tgt_seq_length, embedding_dim)
    
    # Step 4: Split Q_dec, K_dec, V_dec for multi-head attention
    Q_dec = split_heads(Q_dec)
    K_dec = split_heads(K_dec)
    V_dec = split_heads(V_dec)
    
    # Step 5: Scaled Dot-Product Attention for decoder self-attention
    scores_dec = torch.matmul(Q_dec, K_dec.transpose(-2, -1)) / math.sqrt(head_dim)  # Shape: (1, num_heads, tgt_seq_length, tgt_seq_length)
    attn_weights_dec = F.softmax(scores_dec, dim=-1)  # Shape: (1, num_heads, tgt_seq_length, tgt_seq_length)
    attn_output_dec = torch.matmul(attn_weights_dec, V_dec)  # Shape: (1, num_heads, tgt_seq_length, head_dim)
    
    # Step 6: Concatenate heads
    attn_output_dec = attn_output_dec.transpose(1,2).contiguous().view(1, tgt_tensor.size(1), -1)  # Shape: (1, tgt_seq_length, embedding_dim)
    
    # Step 7: Apply output linear layer
    attn_output_dec = out_linear(attn_output_dec)  # Shape: (1, tgt_seq_length, embedding_dim)
    
    # Step 8: Add residual connection and apply layer normalization
    tgt_embeddings = layer_norm1(tgt_embeddings + attn_output_dec)  # Shape: (1, tgt_seq_length, embedding_dim)
    
    # =========================
    # Encoder-Decoder Attention
    # =========================
    
    # Step 1: Compute Q from decoder embeddings and K, V from encoder embeddings
    Q_enc_dec = Q_linear(tgt_embeddings)  # Shape: (1, tgt_seq_length, embedding_dim)
    K_enc_dec = K_linear(src_embeddings)  # Shape: (1, src_seq_length, embedding_dim)
    V_enc_dec = V_linear(src_embeddings)  # Shape: (1, src_seq_length, embedding_dim)
    
    # Step 2: Split Q_enc_dec, K_enc_dec, V_enc_dec for multi-head attention
    Q_enc_dec = split_heads(Q_enc_dec)
    K_enc_dec = split_heads(K_enc_dec)
    V_enc_dec = split_heads(V_enc_dec)
    
    # Step 3: Scaled Dot-Product Attention for encoder-decoder attention
    scores_enc_dec = torch.matmul(Q_enc_dec, K_enc_dec.transpose(-2, -1)) / math.sqrt(head_dim)  # Shape: (1, num_heads, tgt_seq_length, src_seq_length)
    attn_weights_enc_dec = F.softmax(scores_enc_dec, dim=-1)  # Shape: (1, num_heads, tgt_seq_length, src_seq_length)
    attn_output_enc_dec = torch.matmul(attn_weights_enc_dec, V_enc_dec)  # Shape: (1, num_heads, tgt_seq_length, head_dim)
    
    # Step 4: Concatenate heads
    attn_output_enc_dec = attn_output_enc_dec.transpose(1,2).contiguous().view(1, tgt_tensor.size(1), -1)  # Shape: (1, tgt_seq_length, embedding_dim)
    
    # Step 5: Apply output linear layer
    attn_output_enc_dec = out_linear(attn_output_enc_dec)  # Shape: (1, tgt_seq_length, embedding_dim)
    
    # Step 6: Add residual connection and apply layer normalization
    tgt_embeddings = layer_norm1(tgt_embeddings + attn_output_enc_dec)  # Shape: (1, tgt_seq_length, embedding_dim)
    
    # =========================
    # Feed Forward Network
    # =========================
    
    # Step 1: Apply first linear layer
    ff_output = ff_linear1(tgt_embeddings)  # Shape: (1, tgt_seq_length, 64)
    
    # Step 2: Apply ReLU activation
    ff_output = ff_relu(ff_output)  # Shape: (1, tgt_seq_length, 64)
    
    # Step 3: Apply second linear layer
    ff_output = ff_linear2(ff_output)  # Shape: (1, tgt_seq_length, embedding_dim)
    
    # Step 4: Apply dropout
    ff_output = ff_dropout(ff_output)  # Shape: (1, tgt_seq_length, embedding_dim)
    
    # Step 5: Add residual connection and apply layer normalization
    tgt_embeddings = layer_norm2(tgt_embeddings + ff_output)  # Shape: (1, tgt_seq_length, embedding_dim)
    
    # =========================
    # Final Output Layer
    # =========================
    
    # Initialize final output linear layer
    fc_out = nn.Linear(16, 9 + 1)  # embedding_dim=16, vocab_size_decoder=9 (+1 for padding '0')
    
    # Step 1: Apply final linear layer to get logits
    output_logits = fc_out(tgt_embeddings)  # Shape: (1, tgt_seq_length, 10)
    
    # Step 2: Prevent the model from predicting '0' by setting its logit to -inf
    output_logits[:, :, 0] = -float('inf')
    
    # Step 3: Apply softmax to get probabilities
    output_probs = F.softmax(output_logits, dim=-1)  # Shape: (1, tgt_seq_length, 10)
    
    # Step 4: Get predicted token IDs by taking the argmax
    predicted_ids = torch.argmax(output_probs, dim=-1)  # Shape: (1, tgt_seq_length)
    print("Predicted Token IDs:", predicted_ids)
    
    # Step 5: Convert predicted token IDs back to words
    predicted_sentence = numbers_to_sentence_decoder(predicted_ids.squeeze().tolist())
    print("Predicted Sentence:", predicted_sentence)

# =========================
# Main Function
# =========================

def main():
    # Get encoding utilities
    sentence_to_numbers_encoder, numbers_to_sentence_decoder = encoding_utilities()
    
    # Define vocabulary sizes
    vocab_size_encoder = 5  # "the", "cat", "sat", "on", "mat"
    vocab_size_decoder = 9  # "it", "sounds", "like", "you're", "quoting", "a", "classic", "simple", "sentence"
    
    # Get embedding layers
    embedding_encoder, embedding_decoder = embedding_layers(vocab_size_encoder, vocab_size_decoder, embedding_dim=16)
    
    # Create positional encoding
    pe = positional_encoding(embedding_dim=16, max_len=10)
    
    # Initialize transformer components
    components = transformer_components(embedding_dim=16, num_heads=2)
    
    # Define input and output sentences
    encoder_sentence = "The cat sat on the mat."
    decoder_sentence = "It sounds like you're quoting a classic simple sentence!"
    
    # Prepare input and output tensors
    src_tensor, tgt_tensor = prepare_input_output(sentence_to_numbers_encoder, encoder_sentence, decoder_sentence)
    
    # Perform forward pass
    forward_pass(src_tensor, tgt_tensor, embedding_encoder, embedding_decoder, pe, components, numbers_to_sentence_decoder)

if __name__ == "__main__":
    main()
