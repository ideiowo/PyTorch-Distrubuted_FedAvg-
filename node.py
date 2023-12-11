import zmq
import pickle

class Node:
    def __init__(self, role, server_host='127.0.0.1', pub_port=12300, rep_port=12301):
        self.context = zmq.Context()
        self.role = role  # 'server' or 'client'
        self.server_host = server_host
        self.pub_port = pub_port
        self.rep_port = rep_port
        self.pub_socket = None
        self.rep_socket = None

        # 创建 ZeroMQ 套接字
        if self.role == 'client':
            self.req_socket = self.context.socket(zmq.REQ)

    def setup(self):
        # 设置发布者和回复者套接字
        if self.role == 'server':
            self.pub_socket = self.context.socket(zmq.PUB)
            self.pub_socket.bind(f"tcp://{self.server_host}:{self.pub_port}")

            self.rep_socket = self.context.socket(zmq.REP)
            self.rep_socket.bind(f"tcp://{self.server_host}:{self.rep_port}")
        elif self.role == 'client':
            self.pub_socket = self.context.socket(zmq.SUB)
            self.pub_socket.connect(f"tcp://{self.server_host}:{self.pub_port}")
            self.pub_socket.setsockopt_string(zmq.SUBSCRIBE, '')

            self.rep_socket = self.context.socket(zmq.REQ)
            self.rep_socket.connect(f"tcp://{self.server_host}:{self.rep_port}")

    def wait_for_clients(self, expected_clients):
        if self.role != 'server':
            raise ValueError("This method should only be called by the server.")
        
        connected_clients = 0
        while connected_clients < expected_clients:
            # 接收客户端消息
            message = self.rep_socket.recv_string()
            if message == "connect":
                # 发送确认响应
                self.rep_socket.send_string("ack")
                connected_clients += 1
                print(f"Connected clients: {connected_clients}/{expected_clients}")


    def connect_to_server(self):
        if self.role == 'client':
            self.req_socket.connect(f"tcp://{self.server_host}:{self.rep_port}")
            print(f"Client connected to server at {self.server_host}:{self.rep_port}")
            # 发送连接请求
            self.req_socket.send_string("connect")
            # 等待服务端确认
            response = self.req_socket.recv_string()
            print(f"Received response from server: {response}")


    def wait_for_updated_model(self):
        if self.role == 'client':
            print("Client waiting for global model from server...")
            try:
                # 确保套接字已经初始化和连接
                if not self.pub_socket:
                    raise RuntimeError("PUB socket not initialized or connected.")
                
                # 接收服务端发送的消息
                message = self.pub_socket.recv()
                if not message:
                    raise RuntimeError("Received empty response from server.")
                
                # 反序列化消息以获取模型状态
                global_model_state = pickle.loads(message)
                print("Received global model from server.")
                return global_model_state

            except Exception as e:
                print(f"Error receiving global model: {e}")
                return None

    def broadcast_global_model(self, model_state_dict):
        if self.role != 'server':
            raise ValueError("Broadcast method should only be called by the server.")

        try:
            serialized_model = pickle.dumps(model_state_dict)
            self.pub_socket.send(serialized_model)
            print("Broadcasted model to clients.")
        except Exception as e:
            print(f"Error broadcasting model: {e}")


    def broadcast_initial_global_state(self, global_model_state, optimizer_state, criterion_state, rounds, batch_size, epochs, num_clients):
        if self.role != 'server':
            raise ValueError("Broadcast method should only be called by the server.")

        combined_state = {
            'model': global_model_state,
            'parameters': {
                'ROUNDS': rounds,
                'BATCH_SIZE': batch_size,
                'EPOCHS': epochs,
                'NUM_CLIENTS': num_clients
            },
            'optimizer': optimizer_state,
            'criterion': criterion_state
        }

        try:
            serialized_state = pickle.dumps(combined_state)
            self.pub_socket.send(serialized_state)
            print("Broadcasted global model and parameters to clients.")
        except Exception as e:
            print(f"Error broadcasting global state: {e}")
    
    def wait_for_global_state(self):
        if self.role != 'client':
            raise ValueError("This method should only be called by the client.")

        print("Client waiting for global state from server...")

        try:
            # 接收服务端发送的消息
            message = self.pub_socket.recv()
            combined_state = pickle.loads(message)
            print("Received global state from server.")
            return combined_state
        except Exception as e:
            print(f"Error receiving global state: {e}")
            return None



    def send_gradients(self, gradients):
        if self.role != 'client':
            raise ValueError("This method should only be called by the client.")

        # 序列化梯度
        serialized_gradients = pickle.dumps(gradients)

        # 发送梯度到服务端
        try:
            self.req_socket.send(serialized_gradients)
            # 等待服务端的响应
            ack = self.req_socket.recv_string()
            print(f"Received acknowledgement from server: {ack}")
        except Exception as e:
            print(f"Error sending gradients to server: {e}")

    def recv_gradients(self, num_clients):
        if self.role != 'server':
            raise ValueError("This method should only be called by the server.")

        all_gradients = []
        for client_id in range(num_clients):
            try:
                # 接收来自客户端的梯度
                serialized_gradients = self.rep_socket.recv()
                gradients = pickle.loads(serialized_gradients)
                all_gradients.append(gradients)

                # 打印接收到的梯度信息
                print(f"Received gradients from client {client_id + 1}")

                # 发送确认消息给客户端
                self.rep_socket.send_string("ack")
            except Exception as e:
                print(f"Error receiving gradients from client {client_id + 1}: {e}")

        return all_gradients

    def send_model(self, model_state_dict):
        if self.role != 'client':
            raise ValueError("This method should only be called by the client.")

        try:
            # 序列化模型状态
            serialized_model = pickle.dumps(model_state_dict)
            # 发送模型状态到服务端
            self.req_socket.send(serialized_model)
            # 等待服务端的响应
            ack = self.req_socket.recv_string()
            print(f"Received acknowledgement from server: {ack}")
        except Exception as e:
            print(f"Error sending model to server: {e}")


    def recv_model(self, num_clients):
        if self.role != 'server':
            raise ValueError("This method should only be called by the server.")

        all_models = []
        for client_id in range(num_clients):
            try:
                # 接收来自客户端的模型
                serialized_model = self.rep_socket.recv()
                model_state_dict = pickle.loads(serialized_model)
                all_models.append(model_state_dict)

                # 打印接收到的模型信息
                print(f"Received model from client {client_id + 1}")
                # 发送确认消息给客户端
                self.rep_socket.send_string("ack")
            except Exception as e:
                print(f"Error receiving model from client {client_id + 1}: {e}")

        return all_models


    def close(self):
        # 关闭套接字和上下文
        if self.pub_socket:
            self.pub_socket.close()
        if self.rep_socket:
            self.rep_socket.close()
        self.context.term()