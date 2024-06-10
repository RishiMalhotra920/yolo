import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

# Define a model
model = torch.nn.Linear(10, 2)

# Optimizer with a low starting learning rate
initial_lr = 0.001
optimizer = optim.SGD(model.parameters(), lr=initial_lr)

# Target learning rate and the epoch at which it should be reached
target_lr = 0.01
growth_epochs = 10

# Calculate the growth factor per epoch
growth_factor = (target_lr / initial_lr) ** (1 / growth_epochs)


def lr_lambda(epoch):
    if epoch < growth_epochs:
        return growth_factor ** epoch
    else:
        return growth_factor ** growth_epochs


# Create a LambdaLR scheduler
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# Training loop
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
    print(f"Epoch {epoch+1}: lr = {scheduler.get_last_lr()[0]}")
