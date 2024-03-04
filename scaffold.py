from requirements import *
from model import *
from adds import *

class Scaffold:
    def __init__(self, options):
        # self.data = options["Data"]
        self.C = options["C"]
        self.E = options["E"]
        self.B = options["B"]
        self.K = options["K"]
        self.r = options["r"]
        self.training = options["training_data"]
        self.validation = options["validation_data"]
        # print('RRRR: ', self.r)
        self.input_dim = options["input_dim"]
        self.lr = options["lr"]
        self.model = NN(input_dim=self.input_dim  , name='server', B=self.B, E=self.E, lr=self.lr).to(device)
        
        for k, v in self.model.named_parameters():
            self.model.control[k] = torch.zeros_like(v.data)
            self.model.delta_control[k] = torch.zeros_like(v.data)
            self.model.delta_y[k] = torch.zeros_like(v.data)
        self.models = []
        
        for i in range(self.K):
            temp = copy.deepcopy(self.model)
            temp.name = i #self.clients[i]
            temp.control = copy.deepcopy(self.model.control) # c_i
            temp.delta_control = copy.deepcopy(self.model.delta_control)
            temp.delta_y = copy.deepcopy(self.model.delta_y)
            self.models.append(temp)
    
    
    def loss_vals(self, model, dataset):
        loss_f = nn.CrossEntropyLoss().to(device)
        loss = 0
        for idx, (x, y) in enumerate(dataset):
            # print("TYPE OF X: ", (x[0][0][0][0].dtype), x[0][0][0][0])
            x = x.type(torch.float32)
            # print("type y: ", (y[0].dtype), y[0])
            prediction=model(x)
            loss+=loss_f(prediction, y)
            
        loss/=idx+1
        return loss

    
    def accuracy_vals(self, model, dataset):
        """Compute the accuracy of the model on the datatset"""
        correct=0
        
        for x, y in iter(dataset):    
            # print("THE TYPE OF X: ", type(x))
            predictions=model(x)
            pred = torch.max(predictions,1)[1] 
            correct += (pred == y).sum().item()
        
        accuracy = 100*correct/len(dataset.dataset)
        
        return accuracy
            
    def server(self):
        loss_hist=[[(self.loss_vals(self.model, dl).detach()) 
            for dl in self.training]]
        acc_hist=[[self.accuracy_vals(self.model, dl) for dl in self.validation]]
        print("loss_hist: ", loss_hist)
        print("acc_list: ", acc_hist)
        for t in tqdm(range(self.r)):
            print("Round ", t + 1, ":")
            
            m = np.max([int(self.C * self.K), 1])
            index = random.sample(range(0, self.K), m)
            print("index: ", index)
#             index = [0]
            print("index: ", index)
            self.dispatch(index)

            self.client_update(index)

            self.aggregate(index)
            
            loss_hist+=[[float(self.loss_vals(self.model, dl).detach()) 
                for dl in self.training]]
            acc_hist+=[[self.accuracy_vals(self.model, dl) for dl in self.validation]]
            print("loss_hist: ", loss_hist)
            print("acc_list: ", acc_hist)
        return self.model, acc_hist, loss_hist
        
    def aggregate(self, index):
        s = 0.
        for j in index:
            s += self.models[j].len

        x = {}
        c = {}

        for k, v in self.models[0].named_parameters():
            x[k] = torch.zeros_like(v.data)
            c[k] = torch.zeros_like(v.data)

        for j in index:
            for k, v in self.models[j].named_parameters():
                x[k] += self.models[j].delta_y[k] / len(index) # averaging
                c[k] += self.models[j].delta_control[k] / len(index)

        # update x and c
        for k, v in self.model.named_parameters():
            v.data += x[k].data # lr = 1
            self.model.control[k].data += c[k].data * (len(index) / self.K)
            
        

    def dispatch(self, index):
        for j in index:
            for old_params, new_params in zip(self.models[j].parameters(), self.model.parameters()):
                old_params.data = new_params.data.clone()
    

    def client_update(self, index):
        for k in index:
            self.models[k], accuracy_list = train(self.models[k], self.model)
            
    def global_test(self, model):
#         model = self.nn
        model.eval()
        c = self.validation
        for client in range(len(c)):
            model.name = client
            test(model, self.training, self.validation)
