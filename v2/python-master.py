import socket
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import math
import os
import logging

# logging 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

FILE_PATH = os.getenv('FILE_PATH', './')
ROUNDS = int(os.getenv('ROUNDS', 10))


# ResNet18 모델 정의
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock_18(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_18, self).__init__()
        self.conv_1 = conv3x3(inplanes, planes, stride)
        self.bn_1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = conv3x3(planes, planes)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


class ResNet_18(nn.Module):
    def __init__(self, block, layers, num_classes, grayscale):
        super(ResNet_18, self).__init__()
        self.inplanes = 64
        in_dim = 1 if grayscale else 3
        self.conv1g = nn.Conv2d(in_dim, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)
        self.bng = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend([block(self.inplanes, planes) for _ in range(1, blocks)])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1g(x)
        x = self.bng(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_full = x.view(x.size(0), -1)
        logits = self.fc(x_full)
        return logits, x_full


def resnet18(num_classes, grayscale):
    return ResNet_18(BasicBlock_18, [2, 2, 2, 2], num_classes=num_classes, grayscale=grayscale)


class ComplexCNN(nn.Module):
    def __init__(self, grayscale):
        super(ComplexCNN, self).__init__()
        in_dim = 1 if grayscale else 3
        self.out_size = 7 if grayscale else 8
        self.conv1_r = nn.Conv2d(
            in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv1_i = nn.Conv2d(
            in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv2_r = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv2_i = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1_r = nn.Linear(self.out_size * self.out_size * 32, 32)
        self.fc1_i = nn.Linear(self.out_size * self.out_size * 32, 32)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x_real = self.conv1_r(x) - self.conv1_i(x)
        x_imag = self.conv1_r(x) + self.conv1_i(x)
        x_real, x_imag = torch.relu(x_real), torch.relu(x_imag)
        x_real, x_imag = self.pool(x_real), self.pool(x_imag)
        x_real = x_real.view(-1, self.out_size * self.out_size * 32)
        x_imag = x_imag.view(-1, self.out_size * self.out_size * 32)
        x_real_fc1 = self.fc1_r(x_real) - self.fc1_i(x_imag)
        x_imag_fc1 = self.fc1_r(x_imag) + self.fc1_i(x_real)
        x_real_fc1, x_imag_fc1 = torch.relu(x_real_fc1), torch.relu(x_imag_fc1)
        combined = torch.cat([x_real_fc1, x_imag_fc1], dim=1)
        output = self.fc2(combined)
        return output


class Master:
    def __init__(self, server_address=('localhost', 9090), num_classes=10, grayscale=False, max_retries=6, retry_interval=10):
        self.server_address = server_address
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        retries = 0
        while retries < max_retries:
            try:
                self.s.connect(self.server_address)
                logger.info("Connected to file-server")
                break
            except ConnectionRefusedError:
                retries += 1
                logger.warning(
                    f"Connection refused, retrying... ({retries}/{max_retries})")
                time.sleep(retry_interval)

        if retries == max_retries:
            raise ConnectionError(
                "Max retries exceeded, could not connect to the file-server.")

        self.model = ComplexCNN(grayscale)
        torch.save(self.model, os.path.join(FILE_PATH, 'global_model.pt'))

    def deploy_weight(self):
        self.s.sendall("start\n".encode('utf-8'))
        logger.info("Deployed global model to clients.")

    def validate(self, test_loader):
        self.model = self.model.to(device)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (x, y) in test_loader:
                x, y = x.to(device), y.to(device)
                out = self.model(x)
                correct += torch.sum(torch.argmax(out, dim=1) == y).item()
                total += x.shape[0]
        accuracy = correct / total
        logger.info(f"Validation Accuracy: {accuracy:.4f}")
        return accuracy

    def average_weight(self, test_loader):
        filename = self.s.recv(1024).decode('utf-8').strip()
        logger.info(f"Received file: {filename}")

        running_avg = None
        client_ids = [1]  # Simulate client models for aggregation

        for client_id in client_ids:
            client_model = torch.load(os.path.join(
                FILE_PATH, f'client_model_{client_id}.pt'))
            running_avg = self.running_model_avg(
                running_avg, client_model, 1 / len(client_ids))

        self.model.load_state_dict(running_avg)
        accuracy = self.validate(test_loader)
        torch.save(self.model, os.path.join(FILE_PATH, 'global_model.pt'))
        self.s.sendall("global_model.pt\n".encode('utf-8'))

    def send_end_message(self):
        self.s.sendall("end".encode('utf-8'))
        logger.info("Sent end message to clients.")

    def running_model_avg(self, current, next_state, scale):
        if current is None:
            current = {key: next_state[key] * scale for key in next_state}
        else:
            for key in current:
                current[key] += next_state[key] * scale
        return current

    def run(self, test_loader, rounds=100):
        logger.info(f"Server started. Running for {rounds} rounds.")
        for t in range(rounds):
            logger.info(f"Starting round {t + 1}")
            self.deploy_weight()
            self.average_weight(test_loader)
        self.send_end_message()
        logger.info("Training completed.")


if __name__ == "__main__":
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"The pod is running on {device}.")

    test_dataset_CIFAR10 = datasets.CIFAR10(
        root='data', train=False, transform=transforms.ToTensor(), download=True)
    test_loader_CIFAR10 = DataLoader(
        dataset=test_dataset_CIFAR10, batch_size=64, shuffle=False)

    master = Master()
    master.run(test_loader_CIFAR10, rounds=ROUNDS)

    end_time = time.time()
    logger.info(f"Execution Time: {end_time - start_time:.2f} seconds")
