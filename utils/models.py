import torch
import torchtext
from transformers import DistilBertModel

class LSTMbaseGloVe(torch.nn.Module):
    def __init__(self, n_layers:int=3, embed_dim:int=300, hidden_dim:int=256, embeddings:str='42B', bidirectionality:bool=True, freeze:bool=False) -> None:
        super().__init__()
        glove_embeddings = torchtext.vocab.GloVe(embeddings, embed_dim)
        self.LSTM = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(glove_embeddings.vectors, freeze=freeze),
            torch.nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectionality, dropout=0.2)
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
    
class GRUBaseGloVe(torch.nn.Module):
    def __init__(self, n_layers:int=3, embed_dim:int=300, hidden_dim:int=256, embeddings:str='42B', bidirectionality:bool=True, freeze:bool=False) -> None:
        super().__init__()
        glove_embeddings = torchtext.vocab.GloVe(embeddings, embed_dim)
        self.GRU = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(glove_embeddings.vectors, freeze=freeze),
            torch.nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first = True, bidirectional=bidirectionality, dropout=0.2)
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
        if isinstance(module, torch.nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.orthogonal_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)

    def forward(self, overview):
        embeddings, _ = self.GRU(overview)
        embeddings = embeddings[:,-1,:]
        output = self.linear(embeddings)
        return output
    
class DistilBertBaseUncased(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(768, 18)
        )
        print("Number of Parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, attn_mask):
        output = self.model(
            input_ids, 
            attention_mask=attn_mask
        )
        output = self.ffn(output.last_hidden_state[:,-1,:])
        return output