import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=9, img_size=16):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * (img_size//4) * (img_size//4), num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def load_digit_model(weights_path, device='cpu', img_size=16):
    model = SimpleCNN(num_classes=9, img_size=img_size)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model