import torch


def train_cls_reg(
    model,
    data_loader,
    optimizer,
    cls_criterion,
    reg_criterion,
    reg_criterion_size,
    DEVICE,
):
    "Train model for classification and regression tasks"
    model.train()
    cls_running_loss = torch.zeros(1)
    reg_running_loss = torch.zeros(reg_criterion_size)

    for inputs, c_labels, r_labels in data_loader:
        inputs, c_labels, r_labels = (
            inputs.to(DEVICE),
            c_labels.to(DEVICE),
            r_labels.to(DEVICE),
        )

        optimizer.zero_grad()

        c_outputs, r_outputs = model(inputs)
        c_loss = cls_criterion(c_outputs, torch.argmax(c_labels, dim=1)) * 0.1
        r_loss = reg_criterion(r_outputs, c_outputs, r_labels, inputs)

        loss = c_loss + sum(r_loss)

        loss.backward()
        optimizer.step()

        cls_running_loss += c_loss.item() * inputs.size(0)
        for i in range(reg_criterion_size):
            reg_running_loss[i] += r_loss[i].item() * inputs.size(0)

    cls_train_loss = cls_running_loss / len(data_loader.dataset)
    reg_train_loss = reg_running_loss / len(data_loader.dataset)

    return cls_train_loss, reg_train_loss


def test_cls_reg(
    model, data_loader, cls_criterion, reg_criterion, reg_criterion_size, DEVICE
):
    "Evaluate model on classification and regression tasks"
    model.eval()
    cls_running_loss = torch.zeros(1)
    reg_running_loss = torch.zeros(reg_criterion_size)

    with torch.no_grad():
        for inputs, c_labels, r_labels in data_loader:
            inputs, c_labels, r_labels = (
                inputs.to(DEVICE),
                c_labels.to(DEVICE),
                r_labels.to(DEVICE),
            )
            c_outputs, r_outputs = model(inputs)

            c_loss = cls_criterion(c_outputs, torch.argmax(c_labels, dim=1)) * 0.1
            r_loss = reg_criterion(r_outputs, c_outputs, r_labels, inputs)

            cls_running_loss += c_loss.item() * inputs.size(0)
            for i in range(reg_criterion_size):
                reg_running_loss[i] += r_loss[i].item() * inputs.size(0)

    cls_test_loss = cls_running_loss / len(data_loader.dataset)
    reg_test_loss = reg_running_loss / len(data_loader.dataset)

    return cls_test_loss, reg_test_loss
