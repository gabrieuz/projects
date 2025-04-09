def train_model_with_validation(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 5
):
    model = model.to(device)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0

        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            epoch_train_loss += loss.item()

            y_pred_class = torch.argmax(y_pred, dim=1)
            epoch_train_acc += accuracy_fn(y, y_pred_class)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_train_loss /= len(train_dataloader)
        epoch_train_acc /= len(train_dataloader)

        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        with torch.inference_mode():
            for X_val, y_val in val_dataloader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_pred = model(X_val)
                loss_val = loss_fn(val_pred, y_val)
                epoch_val_loss += loss_val.item()
                epoch_val_acc += accuracy_fn(y_val, val_pred.argmax(dim=1))

        epoch_val_loss /= len(val_dataloader)
        epoch_val_acc /= len(val_dataloader)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)

        print(f"[Epoch {epoch + 1:02d}/{epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% "
              f"| Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accuracies,
        "val_accs": val_accuracies
    }

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
):
    model.eval()
    model.to(device)

    total_loss = 0
    total_acc = 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            total_acc += accuracy_fn(y, y_pred.argmax(dim=1))

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    return avg_loss, avg_acc