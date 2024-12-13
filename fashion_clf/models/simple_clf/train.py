import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from fashion_clf.models.simple_clf.dataloaders import get_dataloaders
from fashion_clf.models.simple_clf.resnet import ResNet18


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_csv = pd.read_csv(args.train_csv_path)
    test_csv = pd.read_csv(args.test_csv_path)
    train_loader, test_loader = get_dataloaders(train_csv, test_csv)

    model = ResNet18()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = args.num_epoch

    writer = SummaryWriter(log_dir="runs/experiment")

    best_accuracy = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        all_predictions = []
        all_labels = []
        total = 0
        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="weighted"
        )
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )

        writer.add_scalar("Loss/train", running_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy/test", accuracy, epoch)
        writer.add_scalar("Precision/test", precision, epoch)
        writer.add_scalar("Recall/test", recall, epoch)
        writer.add_scalar("F1/test", f1, epoch)

        for i, (p, r, f) in enumerate(zip(class_precision, class_recall, class_f1)):
            writer.add_scalar(f"Class_{i}/Precision", p, epoch)
            writer.add_scalar(f"Class_{i}/Recall", r, epoch)
            writer.add_scalar(f"Class_{i}/F1", f, epoch)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, "
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Model saved with accuracy: {best_accuracy:.4f}")

    writer.close()
