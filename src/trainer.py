from torch import no_grad

class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, x, y):
        for param in self.model.parameters():
            param.grad = None
        output = self.model(*x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def eval(self, x, y):
        with no_grad():
            output = self.model.eval(*x)
            loss = self.criterion(output, y)
        return loss
