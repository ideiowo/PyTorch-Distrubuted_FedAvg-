import torch

class Server:
    def __init__(self, initial_model):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = initial_model.to(device)  # Ensure the model is on the correct device
        
    def aggregate_gradients(self, client_gradients, lr=0.001):
        num_clients = len(client_gradients)

        # 初始化聚合梯度
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        aggregated_gradients = {name: torch.zeros_like(gradient).to(device) for name, gradient in client_gradients[0].items() if gradient.dtype.is_floating_point}

        # 对每个客户端的梯度进行聚合
        for gradients in client_gradients:
            for name, gradient in gradients.items():
                if gradient.dtype.is_floating_point:
                    aggregated_gradients[name] += gradient / num_clients

        # 更新全局模型的权重
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_gradients:
                    param -= lr * aggregated_gradients[name]

    def aggregate_models(self, client_models):
        num_clients = len(client_models)

        # 如果只有一个客户端，直接使用其模型状态
        if num_clients == 1:
            self.global_model.load_state_dict(client_models[0])
        else:
            # 平均聚合多个模型的状态
            avg_model_state = {}
            for key in client_models[0].keys():
                # 检查参数类型
                if isinstance(client_models[0][key], torch.LongTensor) or isinstance(client_models[0][key], torch.cuda.LongTensor):
                    # 对于 Long 类型，先转换为浮点数，平均后再转换回来
                    temp_tensor_list = [client[key].float() for client in client_models]
                    avg_model_state[key] = torch.mean(torch.stack(temp_tensor_list), 0).long()
                else:
                    # 对于其他类型（通常是浮点数），直接计算平均
                    avg_model_state[key] = torch.mean(torch.stack([client[key] for client in client_models]), 0)

            # 更新全局模型状态
            with torch.no_grad():
                for name, param in self.global_model.named_parameters():
                    if name in avg_model_state:
                        param.copy_(avg_model_state[name])




    def aggregate_weights(self, client_weight_diffs, weights=None):
        """
        聚合客户端发来的权重差，并应用到全局模型上。
        """
        self.optimizer.zero_grad()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 如果没有提供权重，则假设每个客户端的权重相等
        if weights is None:
            weights = [1.0 / len(client_weight_diffs)] * len(client_weight_diffs)

        # 聚合权重差
        for weight, weight_diff in zip(weights, client_weight_diffs):
            for name, param in self.global_model.named_parameters():
                # 将权重差数据移至 GPU
                weight_diff_gpu = {k: v.to(device) for k, v in weight_diff.items()}
                param.grad = weight_diff_gpu[name] * weight if name in weight_diff_gpu else torch.zeros_like(param)

        # 更新全局模型
        self.optimizer.step()
