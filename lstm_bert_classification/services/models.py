import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from lstm_bert_classification.utils.common import ModelType

class BertLSTMModel(nn.Module):
    def __init__(
        self,
        bert_model_name: str,
        num_labels: int,
        hidden_size: int = 100,
        dropout: float = 0.1,
        freeze_bert_layers: int = 0,
        use_attention: bool = False,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        
        # Freeze layers
        for name, param in self.bert.named_parameters():
            layer_num = name.split('.')[2] if 'encoder.layer' in name else None
            if layer_num and int(layer_num) < freeze_bert_layers:
                param.requires_grad = False
                
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(self.hidden_size * (2 if bidirectional else 1), 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
            
        self.input_layer_norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.output_layer_norm = nn.LayerNorm(self.hidden_size * (2 if bidirectional else 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size * (2 if bidirectional else 1), num_labels)

    def init_hidden(self, batch_size, device):
        num_dirs = 2 if self.bidirectional else 1
        h0 = torch.zeros(num_dirs, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(num_dirs, batch_size, self.hidden_size, device=device)
        return (h0, c0)

    def forward(self, input_ids, attention_mask, lengths, turn_mask=None, hidden=None):
        batch_size, seq_len, max_length = input_ids.size()

        # Reshape Input để embed từng turn
        input_ids_flat = input_ids.view(batch_size * seq_len, max_length)
        attention_mask_flat = attention_mask.view(batch_size * seq_len, max_length)
        bert_outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        
        # Lấy [CLS]
        bert_hidden = bert_outputs.last_hidden_state[:, 0, :]       # [batch_size * seq_len, 768]
        bert_hidden = self.input_layer_norm(bert_hidden)

        # Reshape lại để đưa qua LSTM
        lstm_input = bert_hidden.view(batch_size, seq_len, -1)      # [batch_size, seq_len, 768]
        
        # Remove seq padding
        pack_input = torch.nn.utils.rnn.pack_padded_sequence(
            lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, lstm_input.device)
        packed_output, hidden = self.lstm(pack_input, hidden)       # [batch_size, seq_len, 100]

        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=seq_len   # [batch_size, seq_len, 100]
        )

        lstm_output = self.output_layer_norm(lstm_output)
        
        if self.use_attention and turn_mask is not None:
            if turn_mask.sum(dim=1).min() > 0:
                attn_scores = self.attention(lstm_output).squeeze(-1)
                attn_scores = attn_scores.masked_fill(~turn_mask, -1e9)
                # Softmax để lấy attention weights
                attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]
                lstm_output = lstm_output * attn_weights
                
        lstm_output = self.dropout(lstm_output)
        logits = self.classifier(lstm_output)

        return logits, hidden


class BertRNNModel(nn.Module):
    def __init__(
        self,
        bert_model_name: str,
        num_labels: int,
        hidden_size: int = 100,
        dropout: float = 0.1,
        freeze_bert_layers: int = 0,
        use_attention: bool = False,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        
        # Freeze layers
        for name, param in self.bert.named_parameters():
            layer_num = name.split('.')[2] if 'encoder.layer' in name else None
            if layer_num and int(layer_num) < freeze_bert_layers:
                param.requires_grad = False
        
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        self.rnn = nn.RNN(
            input_size=self.bert.config.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True
        )
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(self.hidden_size, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        self.input_layer_norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.output_layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0

    def forward(self, input_ids, attention_mask, lengths, turn_mask=None, hidden=None):
        batch_size, seq_len, max_length = input_ids.size()

        # Reshape để đưa qua BERT
        input_ids_flat = input_ids.view(batch_size * seq_len, max_length)
        attention_mask_flat = attention_mask.view(batch_size * seq_len, max_length)
        bert_outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        
        # Lấy [CLS]
        bert_hidden = bert_outputs.last_hidden_state[:, 0, :]       # [batch_size * seq_len, 768]
        bert_hidden = self.input_layer_norm(bert_hidden)

        # Reshape lại để đưa qua RNN
        rnn_input = bert_hidden.view(batch_size, seq_len, -1)       # [batch_size, seq_len, 768]

        pack_input = torch.nn.utils.rnn.pack_padded_sequence(
            rnn_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        if hidden is None:
            hidden = self.init_hidden(batch_size, rnn_input.device)
        
        packed_output, hidden = self.rnn(pack_input, hidden)        # [batch_size, seq_len, 100]

        rnn_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=seq_len   # [batch_size, seq_len, 100]
        ) 
        
        rnn_output = self.output_layer_norm(rnn_output)
        
        if self.use_attention and turn_mask is not None:
            if turn_mask.sum(dim=1).min() > 0:
                attn_scores = self.attention(rnn_output).squeeze(-1)
                attn_scores = attn_scores.masked_fill(~turn_mask, -1e9)
                # Softmax để lấy attention weights
                attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
                rnn_output = rnn_output * attn_weights
                
        rnn_output = self.dropout(rnn_output)
        logits = self.classifier(rnn_output)

        return logits, hidden


class ModelFactory:
    @staticmethod
    def initialize_model(model_type: str, **kwargs) -> nn.Module:
        model_dict = {
            ModelType.LSTM: BertLSTMModel,
            ModelType.RNN: BertRNNModel,
        }
        if model_type not in model_dict:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model_dict[model_type](**kwargs)
