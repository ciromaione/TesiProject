import zmq


class Socket:
    def __init__(self, socket_type):
        self.socket = zmq.Context().socket(socket_type)

    def send(self, msg):
        self.socket.send_pyobj(msg)

    def recv(self):
        return self.socket.recv_pyobj()

    def send_wait(self, msg):
        self.send(msg)
        return self.recv()


class ServerSocket(Socket):
    def __init__(self, local_port):
        super().__init__(zmq.REP)
        self.socket.bind(f"tcp://*:{local_port}")


class ClientSocket(Socket):
    def __init__(self, server_ip: str, server_port: int):
        super().__init__(zmq.REQ)
        self.socket.connect(f"tcp://{server_ip}:{server_port}")
