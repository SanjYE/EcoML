import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import mobilenet_v2
from codecarbon import EmissionsTracker

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    testset = torchvision.datasets.CIFAR10(root='.', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    model = mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 10)
    model = model.to(device)

    model.load_state_dict(torch.load('mobilenet_v2_cifar10.pth'))  # Modify path as necessary

    tracker = EmissionsTracker(output_dir='emissions/')
    tracker.start()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy on test set: {100 * correct / total}%")

    tracker.stop()
