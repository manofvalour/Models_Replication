import torch
import torch.nn as nn
import torch.nn.functional as F

from model_architecture import Block


##creating out model with torch
class TransformerLanguageModel(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()

        # The input embedding layer
        self.config=config
        self.tokenizer=tokenizer
        vocab_size = tokenizer.vocab_size
        self.tok_embedding_table = nn.Embedding(vocab_size, config.n_embd) ## embedding 
        self.input_position_table = nn.Embedding(config.block_size, config.n_embd) ##input positional encoding

        # The output embedding layer
        self.output_position_table = nn.Embedding(config.block_size, config.n_embd) ## output positional encoding

        # The encoder and decoder blocks
        self.encoder_blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.decoder_blocks = nn.ModuleList([Block(config, is_decoder=True) for _ in range(config.n_layer)])

        # The Linear model
        self.lm_head = nn.Linear(config.n_embd, vocab_size)
        self.lm_head.weight = self.tok_embedding_table.weight     #tying the weight

    def forward(self, idx_input, idx_output, target=None):
        Bx,Tx = idx_input.shape
        By, Ty = idx_output.shape

        input_token_embd = self.tok_embedding_table(idx_input) #(B,T,C)
        input_pos_embd = self.input_position_table(torch.arange(Tx, device =self.config.device)) #(T,C)

        output_token_embd = self.tok_embedding_table(idx_output)
        output_pos_embd = self.output_position_table(torch.arange(Ty, device=self.config.device))

        x = input_token_embd + input_pos_embd #(B,T,C)
        enc_out = self.encoder_blocks(x)              # encoder block output

        y = output_token_embd + output_pos_embd #(B,T,C)
        for decoder in self.decoder_blocks:
            y = decoder(x=y, enc_out=enc_out)   # decoder block output

        ## applying the linear model
        logits= self.lm_head(y) ## (B,T, Vocab_size)

        if target == None:
            loss = None

        else:
            B,T,C = logits.shape
            logits = logits.reshape(B*T, C)
            target = target.reshape(B*T)
            loss = F.cross_entropy(logits, target, label_smoothing=0.1, ignore_index=self.tokenizer.pad_token_id)

        return logits, loss

    @torch.no_grad()
    def generate(self, src, max_new_tokens, sos_token_id=1, eos_token_id=2):
        """
        src: [Batch, Src_Seq_Len] (The English sentence)
        max_new_tokens: How many French words to generate
        """
        device = src.device
        B,T = src.shape

        #Encode the source
        input_token_embd = self.tok_embedding_table(src)
        input_pos_embd = self.input_position_table(torch.arange(T, device=device))
        x = input_token_embd + input_pos_embd
        enc_output = self.encoder_blocks(x)

        # 2. Initialize the decoder input with the [SOS] token
        idx = torch.full((B, 1), sos_token_id, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # Only take the last 'block_size' tokens if the sequence gets too long
            idx_cond = idx[:, -self.config.block_size:]

            # Get embeddings for current decoder input
            output_token_embd = self.tok_embedding_table(idx_cond)
            output_pos_embd = self.output_position_table(torch.arange(idx_cond.shape[1], device=device))
            y = output_token_embd + output_pos_embd

            # Pass through decoder blocks
            decoder_input = y
            for decoder_block in self.decoder_blocks:
                decoder_input = decoder_block(x=decoder_input, enc_out=enc_output)
            decoder_output= decoder_input

            # Apply linear head to get logits
            logits = self.lm_head(decoder_output)

            logits = logits[:, -1, :] # [Batch, Vocab_Size] Focus on the last time step

            idx_next = torch.argmax(logits, dim=-1, keepdim=True) # [Batch, 1]
            idx = torch.cat((idx, idx_next), dim=1) #Append to the sequence

            if (idx_next == eos_token_id).all():
                break

        # Decode the generated dataset
        gen_words = self.tokenizer.batch_decode(idx, skip_special_tokens=True) #input tokens

        return gen_words