import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification
import time

class WSDClassifier(nn.Module):
    def __init__(self):
        super(WSDClassifier, self).__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
    
    def compute_accuracy(self, data_loader, device):
        with torch.no_grad():
            correct_pred, num_examples = 0, 0
            for batch_idx, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_labels = torch.argmax(logits, 1)
                num_examples += labels.size(0)
                correct_pred += (predicted_labels == labels).sum()
            return correct_pred.float() / num_examples * 100
        
    def train_model(self, train_loader, valid_loader, test_loader, device, num_epochs=3, learning_rate=5e-5):
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        start_time = time.time()

        for epoch in range(num_epochs):
            self.model.train()

            for batch_idx, batch in enumerate(train_loader):

                #### Move the data to the currently selected device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                if not batch_idx % 250:
                    print(f'Epoch: {epoch+1:04d}/{num_epochs:04d} | '
                          f'Batch {batch_idx:04d}/{len(train_loader):04d} | '
                          f'Loss: {loss:.4f}')

            self.model.eval()

            with torch.set_grad_enabled(False):
                print(f'Training accuracy: {self.compute_accuracy(train_loader, device):.2f}%\n'
                      f'Valid accuracy: {self.compute_accuracy(valid_loader, device):.2f}%')
            
            print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')

        print(f'Total Training Time: {(time.time() - start_time) / 60:.2f} min')
        print(f'Test accuracy: {self.compute_accuracy(test_loader, device):.2f}%')

if __name__ == '__main__':
    # Example Usage - Just an example, the model will not train.

    # Initialize the model
    model = WSDClassifier()
    # Train the model
    model.train_model(train_loader, valid_loader, test_loader, DEVICE)