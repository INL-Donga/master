import logging
import socket
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os

FILE_PATH = os.getenv('FILE_PATH')
ROUNDS = int(os.getenv('ROUNDS'))

# FILE_PATH = "D:\\INL\\RnD\\middle-controller\\middle-controller\\middle-controller\\parameters\\"
# ROUNDS = 20

# logging 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger()


# ComplexCNN 정의
class ComplexCNN(nn.Module):
    def __init__(self, grayscale):
        if grayscale:
            in_dim = 1
            self.out_size = 7
        else:
            in_dim = 3
            self.out_size = 8
        super(ComplexCNN, self).__init__()

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

        x_real = torch.relu(x_real)
        x_imag = torch.relu(x_imag)

        x_real = self.pool(x_real)
        x_imag = self.pool(x_imag)

        x_real_2 = self.conv2_r(x_real) - self.conv2_i(x_imag)
        x_imag_2 = self.conv2_r(x_imag) + self.conv2_i(x_real)

        x_real_2 = torch.relu(x_real_2)
        x_imag_2 = torch.relu(x_imag_2)

        x_real_2 = self.pool(x_real_2)
        x_imag_2 = self.pool(x_imag_2)

        x_real_2 = x_real_2.view(-1, self.out_size * self.out_size * 32)
        x_imag_2 = x_imag_2.view(-1, self.out_size * self.out_size * 32)

        x_real_fc1 = self.fc1_r(x_real_2) - self.fc1_i(x_imag_2)
        x_imag_fc1 = self.fc1_r(x_imag_2) + self.fc1_i(x_real_2)

        x_real_fc1 = torch.relu(x_real_fc1)
        x_imag_fc1 = torch.relu(x_imag_fc1)

        combined = torch.cat([x_real_fc1, x_imag_fc1], dim=1)

        output = self.fc2(combined)

        return output


# Master 클래스 정의
class Master:
    def __init__(self, server_address=('localhost', 9090), num_classes=10, grayscale=False, max_retries=6, retry_interval=10):
        self.server_address = server_address
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 재시도 로직 추가
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
            logger.error(
                "Max retries exceeded, could not connect to the file-server.")
            raise ConnectionError(
                "Max retries exceeded, could not connect to the file-server.")

        self.model = ComplexCNN(grayscale)
        torch.save(self.model.state_dict(), os.path.join(
            FILE_PATH, 'global_model.pt'))

    def deploy_weight(self):
        startmsg = "start\n"
        self.s.sendall(startmsg.encode('utf-8'))
        logger.info("Weights deployed.")

    def validate(self, test_loader):
        self.model = self.model.to(device)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (x, y) in test_loader:
                x = x.to(device)
                y = y.to(device)
                out = self.model(x)
                correct += torch.sum(torch.argmax(out, dim=1) == y).item()
                total += x.shape[0]
        return correct / total

    def average_weight(self, test_loader):
        filename = self.s.recv(1024).decode('utf-8')
        logger.info(f"Received file: {filename}")

        running_avg = None
        train_client_id = [1]

        for k in train_client_id:
            client_model = torch.load(os.path.join(
                FILE_PATH, f'client_model_{k}.pt'))
            running_avg = self.running_model_avg(
                running_avg, client_model, 1 / len(train_client_id))

        state_dict = self.model.state_dict()
        state_dict.update(running_avg)
        self.model.load_state_dict(state_dict)

        accuracy = self.validate(test_loader)
        logger.info(f"Validation accuracy: {accuracy}")
        torch.save(self.model.state_dict(), os.path.join(
            FILE_PATH, 'global_model.pt'))

        filename = "global_model.pt\n"
        self.s.sendall(filename.encode('utf-8'))

    def send_end_message(self):
        msg = "end"
        logger.info("Sending end message to server.")
        self.s.sendall(msg.encode('utf-8'))

    def running_model_avg(self, current, next, scale):
        if current is None:
            current = {key: next[key] * scale for key in next}
        else:
            for key in current:
                current[key] += next[key] * scale
        return current

    def run(self, test_loader, rounds=100):
        logger.info(f"Server IP and port: {self.s.getpeername()}")
        logger.info(f"Local IP and port: {self.s.getsockname()}")

        for t in range(rounds):
            logger.info(f"Starting Round: {t + 1}")
            self.deploy_weight()
            self.average_weight(test_loader)

        self.send_end_message()


if __name__ == "__main__":

    start = time.time()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"The pod is running with {device}")

    test_dataset_CIFAR10 = datasets.CIFAR10(
        root='data', train=False, transform=transforms.ToTensor(), download=True)
    test_loader_CIFAR10 = DataLoader(
        dataset=test_dataset_CIFAR10, batch_size=64, shuffle=False)

    master = Master()
    master.run(test_loader_CIFAR10, rounds=ROUNDS)

    end = time.time()
    logger.info(f"Execution time: {end - start:.5f} seconds")
