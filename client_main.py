# client_main.py
import argparse
from client import Client
from utils.data_utils import load_iid_data, load_non_iid_data
from models.architecture import ResNet18,CNN
import torch.optim as optim
import torch
import node
import json
import torch.nn as nn
import matplotlib.pyplot as plt

# 解析命令行參數
parser = argparse.ArgumentParser()
parser.add_argument('client_id', type=int, help='Client ID')
args = parser.parse_args()

CLIENT_ID = args.client_id


# 初始化模型和優化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 創建客戶端節點
client_node = node.Node(role='client')
client_node.setup()  # 初始化並設置套接字
# 連接到服務端
client_node.connect_to_server()


# 實例化模型並加載全局模型狀態
initial_model = CNN().to(device)  # 確保模型在正確的設備上
optimizer = optim.Adam(initial_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 客戶端等待接收來自服務端的全局狀態
global_state = client_node.wait_for_global_state()
if global_state:
    # 更新模型狀態
    global_model_state = global_state['model']
    initial_model.load_state_dict(global_model_state)
    # 更新優化器狀態
    if 'optimizer' in global_state:
        optimizer_state = global_state['optimizer']
        optimizer.load_state_dict(optimizer_state)

    # 更新損失函數狀態
    if 'criterion' in global_state:
        criterion_state = global_state['criterion']
        criterion.load_state_dict(criterion_state)

    # 更新其他參數（如 ROUNDS, BATCH_SIZE 等）
    if 'parameters' in global_state:
        parameters = global_state['parameters']
        ROUNDS = parameters.get('ROUNDS')
        BATCH_SIZE = parameters.get('BATCH_SIZE')
        EPOCHS = parameters.get('EPOCHS')
        NUM_CLIENTS = parameters.get('NUM_CLIENTS')
else:
    print("Failed to receive global state from server.")
    exit(1)


# 加載數據
client_loaders, client_val_loaders = load_iid_data(num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE)

# 獲取客戶端的數據加載器
data_loader = client_loaders[CLIENT_ID]
val_loader = client_val_loaders[CLIENT_ID]
# 創建客戶端實例
client = Client(client_id=CLIENT_ID, 
                data_loader=data_loader, 
                model=initial_model,
                optimizer=optimizer,
                criterion=criterion)

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
# 客戶端訓練和驗證

import copy

# 初始化存储结果的列表
train_losses = []
val_losses = []
accuracies = []

for round in range(ROUNDS):
    # 保存訓練前的模型狀態
    pre_train_model = copy.deepcopy(client.model)
    gradients, train_loss, val_loss, accuracy = client.train_and_validate(epochs=EPOCHS, val_loader=val_loader)
        # 更新列表
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    accuracies.append(accuracy)

    #client_node.send_model(client.model.state_dict())
    client_node.send_gradients(gradients)
    #after_train_model = copy.deepcopy(client.model)
    
    # 等待服務端發送更新後的全局模型
    updated_model_state = client_node.wait_for_updated_model()
    if updated_model_state is None:
        print(f"Failed to receive updated model from server for round {round + 1}.")
        continue  # 如果沒有接收到模型，繼續等待或重試
    client.set_model(updated_model_state)

    #model_diff = calculate_model_diff(after_train_model, client.model)
    #print(f"Global_Client difference at round {round + 1}: {model_diff}")
    

    print(f"Updated model received and loaded for round {round + 1}.")
# 绘制图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 绘制训练和验证损失
ax1.plot(range(1, ROUNDS + 1), train_losses, label='Train Loss')
ax1.plot(range(1, ROUNDS + 1), val_losses, label='Validation Loss')
ax1.set_title('Train and Validation Loss')
ax1.set_xlabel('Rounds')
ax1.set_ylabel('Loss')
ax1.legend()

# 绘制准确率
ax2.plot(range(1, ROUNDS + 1), accuracies, label='Accuracy', color='orange')
ax2.set_title('Accuracy')
ax2.set_xlabel('Rounds')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()

plt.tight_layout()
plt.show()