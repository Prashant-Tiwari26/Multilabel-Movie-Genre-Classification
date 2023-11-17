import torch
import torchtext

class LSTMbaseGloVe(torch.nn.Module):
    def __init__(self, n_layers:int=3, embed_dim:int=300, hidden_dim:int=256, embeddings:str='42B', bidirectionality:bool=True, freeze:bool=False) -> None:
        super().__init__()
        glove_embeddings = torchtext.vocab.GloVe(embeddings, embed_dim)
        self.LSTM = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(glove_embeddings.vectors, freeze=freeze),
            torch.nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectionality)
        )
        self.linear = torch.nn.LazyLinear(18)

    def forward(self, overview):
        embeddings, _ = self.LSTM(overview)
        embeddings = embeddings[:,-1,:]
        output = self.linear(embeddings)
        return output