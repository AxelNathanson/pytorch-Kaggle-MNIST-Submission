import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.float)

# Clean up old comments, write descriptions to all functions.
# Fix the overtrain-function.


class Solver(object):
    def __init__(self, model, train_set, validation_set, test_set, **kwargs):
        self.model = model
        self.train_set = train_set
        self.validation_set = validation_set

        self.test_loader = DataLoader(test_set, batch_size=1)

        # Optional arguments
        learning_rate = kwargs.pop('learning_rate', 1e-4)
        weight_decay = kwargs.pop('weight_decay', 3e-4)
        self.print_every = kwargs.pop('printevery', 175)
        self.batch_size = kwargs.pop('batch_size', 64)
        lr_rate_decay = kwargs.pop('lr_rate_decay', 0.5)
        lr_decay_epochs = kwargs.pop('decay_every_', 1)

        self.criteria = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=learning_rate,
                                          weight_decay= weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                         step_size=lr_decay_epochs, 
                                                         gamma=lr_rate_decay)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Reset all of my saved history
        self._reset()
        self.epoch_acc = None

    def _reset(self):
        """Resets all the history variables of the training.
        """        
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.train_acc_history = []
    
    def _step(self, image, target):
        """Performs one step of the function, given an image and target
        
        Arguments:
            image {tensor} -- Input to model
            target {array} -- Target number
        """     
        # Forward pass
        output = self.model(image)
        loss = self.criteria(output, target)
        self.train_loss_history.append(loss.item())
        
        # Accuracy
        _, prediction = torch.max(output, 1)
        self.epoch_acc.append(np.mean((prediction == target).data.cpu().numpy()))

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_epochs = 10):
        print(f'Started training. Will run for: {num_epochs} Epochs.',
              f'Iterations per Epoch: {int(len(self.train_set)/self.batch_size) + 1}.')

        train_loader = DataLoader(self.train_set,
                                  batch_size=self.batch_size, 
                                  shuffle=True)

        validation_loader = DataLoader(self.validation_set, 
                                       batch_size=self.batch_size, 
                                       shuffle=True)

        for epoch in range(num_epochs):
            self.epoch_acc = []
            self.model.train()
            for it, data in enumerate(train_loader):
                if it%self.print_every == 0 and it != 0:
                    print(f'Done with iteration: {it}/{len(train_loader)}.')
                
                image, target = data
                image.to(self.device); target.to(self.device)
             
                self._step(image, target)
            
            self.train_acc_history.append(np.mean(self.epoch_acc))

            self.model.eval()
            val_acc = []
            for batch, data in enumerate(validation_loader):
                image, target = data
                image.to(self.device)
                target.to(self.device)

                output = self.model(image)
                # loss = self.criteria(output, target)
                # self.train_loss_history.append(loss.item())

                # Accuracy
                _, prediction = torch.max(output, 1)
                val_acc.append(np.mean((prediction == target).data.cpu().numpy()))

            self.val_acc_history.append(np.mean(val_acc))

            # Finnish the epoch with updating the lr_rate.
            self.scheduler.step()

            print(f'Epoch:{epoch+1}/{num_epochs}\nLoss: {self.train_loss_history[-1]}',  
                  f'\nValidation accuracy: {self.val_acc_history[-1]}', 
                  f'\nTraining accuracy: {self.train_acc_history[-1]}')

    def test_accuracy(self):
        """
        Evaluate the accuracy on the test-set and saves as submission.
        """        
        from generate_submission import generate_submission
        test_prediction = []

        for image in self.test_loader:
            output = self.model(image)

            _, prediction = torch.max(output, 1)

            test_prediction.append(prediction.item())
        
        generate_submission(test_prediction)
    
        



