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
        if bidirectionality == True:
            self.linear = torch.nn.Linear(2*hidden_dim, 18)
        else:
            self.linear = torch.nn.Linear(hidden_dim, 18)

        self.apply(self._init_weights)
        print("Number of Parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.orthogonal_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)

    def forward(self, overview):
        embeddings, _ = self.LSTM(overview)
        embeddings = embeddings[:,-1,:]
        output = self.linear(embeddings)
        return output