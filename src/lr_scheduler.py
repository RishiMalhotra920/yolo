from torch.optim.lr_scheduler import SequentialLR, ExponentialLR, ConstantLR
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR

# Define the custom learning rate schedule


import torch
from torch.optim.lr_scheduler import _LRScheduler


class CustomExponentialLR(_LRScheduler):
    """
    It increases the learning rate exponentially from 0.001 to 0.01 in 10 epochs 
    and then keeps it constant at 0.01 for the remaining epochs.
    """

    def __init__(self, optimizer, init_lr, final_lr, growth_epochs, last_epoch=-1):
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.growth_epochs = growth_epochs
        self.growth_factor = (final_lr / init_lr) ** (1 / growth_epochs)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # self.base_lrs stores the learning rate for each group in the optimizer.
        # self.last_epoch is the last epoch. it is updated by the base class.
        if self.last_epoch < self.growth_epochs:
            return [base_lr * (self.growth_factor ** self.last_epoch) for base_lr in self.base_lrs]
        return [self.final_lr for base_lr in self.base_lrs]


def get_custom_lr_scheduler(optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    """
    It increases the learning rate exponentially from 0.001 to 0.01 in 10 epochs
    and then keeps it constant at 0.01 for the remaining epochs.
    """
    scheduler = CustomExponentialLR(
        optimizer, init_lr=0.001, final_lr=0.005, growth_epochs=10)
    return scheduler


def get_fixed_lr_scheduler(optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1)


if __name__ == '__main__':
    # Test the learning rate schedule above
    model = torch.nn.Linear(10, 2)
    initial_lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)
    # scheduler = get_fixed_lr_scheduler(optimizer)
    scheduler = get_custom_lr_scheduler(optimizer)

    # Training loop
    learning_rates = []
    for epoch in range(20):  # example for 20 epochs
        # Training code here
        optimizer.zero_grad()
        # Assume some fake loss
        loss = model(torch.randn(1, 10))
        # loss.backward()
        optimizer.step()

        # Update the learning rate
        scheduler.step()

        # Print learning rate
        learning_rates.append(scheduler.get_last_lr()[0])

    plt.plot(learning_rates)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Custom Exponential Learning Rate Scheduler")
    plt.show()
