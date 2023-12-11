from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import random
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_iid_data(data_dir='./data/CIFAR10/train', num_clients=10, batch_size=32, val_split=0.1):
    # 资料的前处理和转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 从 CIFAR10/train 加载数据
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # 分割资料以模拟多个客户端的环境且确保均匀的数据分布
    data_size = len(dataset)
    client_data_size = data_size // num_clients
    indices = list(range(data_size))
    random.shuffle(indices)

    client_loaders = []
    client_val_loaders = []

    for i in range(num_clients):
        # 确定每个客户端数据的索引
        start_idx = i * client_data_size
        end_idx = start_idx + client_data_size if i != (num_clients - 1) else data_size
        subset_indices = indices[start_idx:end_idx]
        dataset_per_client = Subset(dataset, subset_indices)

        # 创建训练和验证数据集
        val_size = int(len(dataset_per_client) * val_split)
        train_size = len(dataset_per_client) - val_size
        train_dataset, val_dataset = random_split(dataset_per_client, [train_size, val_size])

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        client_loaders.append(train_loader)
        client_val_loaders.append(val_loader)

    return client_loaders, client_val_loaders

def load_non_iid_data(data_dir='./data/CIFAR10/train', num_clients=10, batch_size=32, val_split=0.1, imbalance_factor=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)

    client_loaders = []
    client_val_loaders = []

    # 确保每个客户端都有一些每种类别的数据，但数量不均等
    for i in range(num_clients):
        client_indices = []

        # 分配标签的不平衡数量
        for label, indices in class_indices.items():
            sample_size = int(len(indices) * (1 - random.uniform(0, imbalance_factor)))
            sampled_indices = random.sample(indices, sample_size)
            client_indices.extend(sampled_indices)

        dataset_per_client = Subset(train_dataset, client_indices)
        train_loader = DataLoader(dataset_per_client, batch_size=batch_size, shuffle=True)
        client_loaders.append(train_loader)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        client_val_loaders.append(val_loader)

    return client_loaders, client_val_loaders


def print_loader_details(loader):
    label_count = {}
    for _, labels in loader:
        for label in labels.numpy():
            label_count[label] = label_count.get(label, 0) + 1
    # 對字典按照鍵進行排序
    sorted_label_count = {k: label_count[k] for k in sorted(label_count)}
    return sorted_label_count

if __name__ == "__main__":
    
    client_loaders, client_val_loaders = load_non_iid_data(num_clients=3)
    
    client_train_label_counts = []

    for train_loader in client_loaders:
        train_label_count = print_loader_details(train_loader)
        client_train_label_counts.append(train_label_count)

    # 將數據整理成一個二維數組
    num_labels = 10  # 假設有 10 個不同的類別
    data_matrix = np.zeros((len(client_loaders), num_labels), dtype=int)  # 將 dtype 設置為 int
    
    for i, counts in enumerate(client_train_label_counts):
        for label, count in counts.items():
            data_matrix[i, label] = count

    # 繪製熱圖
    plt.figure(figsize=(10, 5))
    sns.heatmap(data_matrix, annot=True, fmt="g", yticklabels=[f"Client {i+1}" for i in range(len(client_loaders))], xticklabels=range(num_labels))
    plt.xlabel("Data Category")
    plt.ylabel("Client")
    plt.title("Training Data Distribution Across Clients")
    plt.show()