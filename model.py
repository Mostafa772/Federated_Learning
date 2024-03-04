from requirements import *
    
class NN(nn.Module):
    def __init__(self, NFILT=256, NOUT=4, name, B, E, lr) -> None:
        super(NN, self).__init__()
        self.name = name
        self.B = B
        self.E = E
        self.len = 0
        self.lr = lr
        self.loss = 0
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}
        
        super(NN,self).__init__()
        self.conv0 = nn.Conv2d(1,NFILT,kernel_size=(200,3),padding=(0,1),bias=False)
        self.bn0 = nn.BatchNorm2d(NFILT)
        self.gru = nn.GRU(input_size=NFILT,hidden_size=128,num_layers=1,batch_first=True,bidirectional=False)
        self.fc1 = nn.Linear(128,NOUT)


    def forward(self, x):
		x = F.relu(self.bn0(self.conv0(x)))
        x = x.squeeze().permute(0,2,1)
        x,_ = self.gru(x)
        x = F.dropout(x,p=0.5,training=self.training)
        x = self.fc1(x)
        return x
        
        # # x = x.double()
        # return self.net(x)
    

















    # class LeNet5(nn.Module):
    # def __init__(self, input_dim, name, B, E, lr) -> None:
    #     super(LeNet5, self).__init__()
    #     self.name = name
    #     self.B = B
    #     self.E = E
    #     self.len = 0
    #     self.lr = lr
    #     self.loss = 0
    #     self.control = {}
    #     self.delta_control = {}
    #     self.delta_y = {}
        
    #     self.net = nn.Sequential(
    #         nn.Conv2d(1, 6, 5),
    #         nn.ReLU(True),
    #         nn.MaxPool2d(2),
    #         nn.Conv2d(6, 16, 5),
    #         nn.ReLU(True),
    #         nn.MaxPool2d(2),
    #         nn.Flatten(),
    #         nn.Linear(19552, 120),
    #         nn.ReLU(True),
    #         nn.Linear(120, 84),
    #         nn.ReLU(True),
    #         nn.Linear(84, 10),
    #     )
    # def forward(self, x):
    #     # x = x.double()
    #     return self.net(x)
