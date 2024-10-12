import socket
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time


def log_message(message):
    current_time = time.strftime(
        "%M:%S", time.gmtime())  # [mm:ss] 포맷으로 시간 가져오기
    print(f"[{current_time}] {message}")

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
        self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
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


class Master:
    def __init__(self, server_address=('localhost', 9090), num_classes=10, grayscale=False):
        self.server_address = server_address
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(self.server_address)
        self.model = resnet18(num_classes=num_classes, grayscale=grayscale)
        torch.save(
            self.model, 'D:\INL\RnD\middle-controller\middle-controller\middle-controller\parameters\global_model.pt')

    def deploy_weight(self):
        startmsg = "start\n"
        self.s.sendall(startmsg.encode('utf-8'))
        print("deploy")

    def validate(self, test_loader):
        self.model = self.model.to(device)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (x, y) in test_loader:
                x = x.to(device)
                y = y.to(device)
                out, _ = self.model(x)
                correct += torch.sum(torch.argmax(out, dim=1) == y).item()
                total += x.shape[0]
        return correct / total

    def average_weight(self, test_loader):
        filename = self.s.recv(1024).decode('utf-8')
        print(f"Received file: {filename}")

        running_avg = None
        train_client_id = [1]

        for k in train_client_id:
            client_model = torch.load(
                f'D:\INL\RnD\middle-controller\middle-controller\middle-controller\parameters\client_model_{k}.pt')
            running_avg = self.running_model_avg(
                running_avg, client_model.state_dict(), 1 / len(train_client_id))

        self.model.load_state_dict(running_avg)
        accuracy = self.validate(test_loader)
        print(f"Accuracy: {accuracy}")
        torch.save(
            self.model, 'D:\INL\RnD\middle-controller\middle-controller\middle-controller\parameters\global_model.pt')

        filename = "global_model.pt\n"
        self.s.sendall(filename.encode('utf-8'))

    def send_end_message(self):
        msg = "end"
        print(msg)
        self.s.sendall(msg.encode('utf-8'))

    def running_model_avg(self, current, next, scale):
        if current is None:
            current = {key: next[key] * scale for key in next}
        else:
            for key in current:
                current[key] += next[key] * scale
        return current

    def run(self, test_loader, rounds=100):
        print(f"서버 IP 및 포트: {self.s.getpeername()}")
        print(f"내 로컬 IP 및 포트: {self.s.getsockname()}")

        for t in range(rounds):
            print(f"Starting Round: {t + 1}")
            self.deploy_weight()
            self.average_weight(test_loader)

        self.send_end_message()


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_message("The pod running with {0}".format(device))

    # CIFAR-10 데이터셋 정의
    test_dataset_CIFAR10 = datasets.CIFAR10(root='data',
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)

    test_loader_CIFAR10 = DataLoader(dataset=test_dataset_CIFAR10,
                                     batch_size=64,
                                     shuffle=False)

    # Master 클래스 인스턴스 생성 및 실행
    master = Master()
    master.run(test_loader_CIFAR10)
