import torch.nn as nn


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class MultiIntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(MultiIntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)
        self.sigmoid = nn.Sigmoid()
        self.reset_params()

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return self.sigmoid(x)
    
    def reset_params(self):
        nn.init.uniform_(self.linear.weight)
        nn.init.uniform_(self.linear.bias)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.2):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        # self.linear1 = nn.Linear(input_dim, num_slot_labels * 2)
        # self.linear2 = nn.Linear(num_slot_labels * 2, num_slot_labels)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(input_dim, num_slot_labels)
        # self.reset_params()

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
        # return self.linear2(self.activation(self.linear1(x)))
    
    def reset_params(self):
        nn.init.uniform_(self.linear.weight)
        nn.init.uniform_(self.linear.bias)

class IntentTokenClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentTokenClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class TagIntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(TagIntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.softmax(self.linear(x))
        # return self.softmax(self.linear(x))