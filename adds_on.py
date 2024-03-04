from requirements import *
sys.path.append('../')


@torch.no_grad()
def copy_parameters(model, zero=False, model_from=Union[OrderedDict[str, torch.Tensor], torch.nn.Module]):
    if zero :
        for i, key in enumerate(model.state_dict().keys()):
            x = torch.zeros_like(model.state_dict()[key])
            model.state_dict()[key].copy_(torch.nn.Parameter(x)) 
#             print(model.state_dict()[key])
    else:
        if isinstance(model_from, torch.nn.Module):
            assert len(model.state_dict().keys()) == len(model_from.state_dict().keys()), "The models should have the same structure"
            for key in model_from.state_dict().keys():
                model.state_dict()[key].copy_(model_from.state_dict()[key])  
        elif isinstance(model_from, OrderedDict):
            for key in model_from.keys():
                model.state_dict()[key].copy_(model_from[key])  
    return model

def plot_acc_loss(title:str, loss_hist:list, acc_hist:list):
    plt.figure()
    
    plt.suptitle(title)

    plt.subplot(1,2,1)
    lines=plt.plot(loss_hist)
    plt.title("Loss")
    plt.legend(lines,["C1", "C2", "C3"])

    plt.subplot(1,2,2)
    lines=plt.plot(acc_hist )
    plt.title("Accuracy")
    plt.legend(lines, ["C1", "C2", "C3"])




def validation(model, data):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    val_loss = []
    correct = 0
    total = 0
    min_accuracy = 10
    with torch.no_grad():
        for X_val, val_target in data:
            X_val, val_target = X_val.to(device), val_target.to(device)
            y_pred = model(X_val)
            loss = criterion(y_pred, val_target)
            val_loss.append(loss)
            pred = torch.max(y_pred,1)[1]
            predicted_values.append(pred)
            correct += (pred == val_target).sum().item()
            total += len(val_target)
        accuracy = correct * 100 / total
        accuracy_list.append(accuracy)
    
    return accuracy, np.mean(val_loss)


def train(model, server, t_data, val_data):
    accuracy_list = []
    model.train()
    """GET THE DATA"""
    print("model.name: ", model.name)
    train_data = t_data[model.name]
    validation_data = val_data[model.name]
#     test_data = clients_loaders[model.name][2]

    model.len = len("""Training Data""")
    print('training...')
    criterion = nn.CrossEntropyLoss().to(device)
    model_before = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001) #ScaffoldOptimizer(model.parameters(), lr=model.lr, weight_decay=1e-4) #torch.optim.SGD(model.parameters(), lr=0.0001)
#     lr_step = StepLR(optimizer, step_size=10, gamma=0.1)
    min_epochs = 0
    best_model = None
    min_val_loss = 5
    
    for epoch in tqdm(range(model.E)):
        train_loss = []
        for X, target in train_data:
            X, target = X.to(device), target.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, target)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#             optimizer.step(server.control, model.control)
#         lr_step.step()
        # evaluation
        accuracy, val_loss = validation(model, validation_data)
        accuracy_list.append(accuracy)
        print(epoch, min_epochs, val_loss, min_val_loss, len(t_data))
        if (epoch + 1 > min_epochs) and (val_loss < min_val_loss):
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
            
        print('epoch {:03d} Accuracy {} train_loss {:.8f} val_loss {:.8f}'.format(epoch, accuracy,  np.mean(train_loss), np.mean(val_loss)))
        model.train()
        
        
    model = copy.deepcopy(best_model)
    """
    Update c
    
    c+ <- c_i - c + 1/(steps * lr) * (x - y_i)
    
    save model
    """
    temp = {}
    for k, v in model.named_parameters():
        temp[k] = v.data.clone()
        
    for k, v in model_before.named_parameters():
        local_steps = model.E * len(train_data)
        model.control[k] = model.control[k] - server.control[k] + (v.data - temp[k]) / (local_steps * model.lr)
        model.delta_y[k] = temp[k] - v.data
        model.delta_control[k] = model.control[k] - model_before.control[k]
    
    return model, accuracy_list
            
def test(model, training, validation):
    model.eval()
    Dtr, Dte = training[model.name], validation[model.name]
    pred = []
    y = []
    correct=0
    total=0
    predicted_values= []
    for (seq, target) in tqdm(Dte):
        with torch.no_grad():
            seq = seq.to(device)
            y_pred = model(seq)
            pred = torch.max(y_pred,1)[1]

            predicted_values.append(pred)
            correct += (pred == target).sum()
            total += len(target)

#     pred = np.array(pred)
#     y = np.array(y)
    print("Accuracy: ", correct*100/total )
