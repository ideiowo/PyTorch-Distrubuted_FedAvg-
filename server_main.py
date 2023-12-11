from server import Server
from models.architecture import ResNet18,CNN
from torchvision import datasets, transforms
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import torch
import node
import json
import time
import copy
import matplotlib.pyplot as plt

# 從 JSON 文件讀取參數
with open('parameters.json', 'r') as file:
    params = json.load(file)

NUM_CLIENTS = params['NUM_CLIENTS']
ROUNDS = params['ROUNDS']
BATCH_SIZE = params['BATCH_SIZE']
EPOCHS = params['EPOCHS']

# 載入所有客戶端的資料
# 第一步：載入驗證資料
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = datasets.ImageFolder('./data/CIFAR10/validation', transform=val_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定義測試模型的函數
def test_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def calculate_model_diff(model1, model2):
    total_diff = 0
    param_count = 0
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 == name2:
            diff = (param1 - param2).abs().mean().item()
            total_diff += diff
            param_count += 1
    average_diff = total_diff / param_count
    return average_diff
# 初始化服務端節點
server_node = node.Node(role='server')
server_node.setup()
server_node.wait_for_clients(expected_clients=NUM_CLIENTS)  # 假設我們期望有2個客戶端
print("All clients connected. Starting communication...")

# 初始化服務端和客戶端
# 定義運行操作的設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
initial_model = CNN().to(device)
optimizer = optim.Adam(initial_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 創建服務端實例
server = Server(initial_model)

# 廣播優化器和損失函數狀態
# 调用 broadcast_global_state 方法
server_node.broadcast_initial_global_state(global_model_state = initial_model.state_dict(),
                                    optimizer_state = optimizer.state_dict(),
                                    criterion_state = criterion.state_dict(),
                                    rounds = ROUNDS,
                                    batch_size = BATCH_SIZE,
                                    epochs = EPOCHS,
                                    num_clients = NUM_CLIENTS)

test_losses = []
test_accuracies = []

# 收集梯度並進行聚合
for round in range(ROUNDS):
    print(f"Round {round + 1}: Receiving gradients from clients...")
    
    # 接收客戶端的梯度
    
    client_gradients = server_node.recv_gradients(NUM_CLIENTS)
    # 保存聚合前的模型狀態
    #pre_aggregate_model = copy.deepcopy(server.global_model)

    # 聚合權重差並更新模型
    server.aggregate_gradients(client_gradients)

    ## 比較聚合前後的模型狀態
    #server_model_diff = calculate_model_diff(pre_aggregate_model, server.global_model)
    #print(f"Model difference after round {round + 1}: {server_model_diff}")

    # 廣播更新後的全局模型
    server_node.broadcast_global_model(server.global_model.state_dict())
    
    # 使用全局模型對測試資料進行預測，並輸出損失和準確率
    test_loss, test_accuracy = test_model(server.global_model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    print(f"Round {round + 1} - Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%")


# 创建一个图表，包含1行2列的子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 在第一个子图上绘制Test Loss
ax1.plot(range(1, ROUNDS + 1), test_losses, label="Test Loss", color='blue')
ax1.set_title(f'Test Loss over Rounds (Clients: {NUM_CLIENTS}, Epochs: {EPOCHS})')
ax1.set_xlabel('Rounds')
ax1.set_ylabel('Loss')
ax1.legend()

# 在第二个子图上绘制Test Accuracy
ax2.plot(range(1, ROUNDS + 1), test_accuracies, label="Test Accuracy", color='red')
ax2.set_title(f'Test Accuracy over Rounds (Clients: {NUM_CLIENTS}, Epochs: {EPOCHS})')
ax2.set_xlabel('Rounds')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()

# 显示整个图表
plt.show()

#python client_main.py 0
#python server_main.py