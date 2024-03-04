from requirements import * 
from scaffold import *
from adds import *
from dataset import Dataset
# dataset_train= Dataset('./new_data/train/').remove_powerline_noise_class()
# dataset_validate = Dataset('./new_data/validate/').remove_powerline_noise_class()
# print(len(dataset_train))


# NWORKERS = 12
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# TRAIN = DataLoader(dataset=dataset_train,
#                    batch_size=8,
#                    shuffle=True,
#                    drop_last=False,
#                    num_workers=NWORKERS)

# TEST = DataLoader(dataset=dataset_validate,
#                    batch_size=8,
#                    shuffle=True,
#                    drop_last=False,
#                    num_workers=NWORKERS)


dataset_train= Dataset('./new_data/train/').remove_powerline_noise_class()
dataset_validate = Dataset('./new_data/validate/').remove_powerline_noise_class()

# print(type(dataset_train[0][0][0][0][0]), dataset_train[0][0][0][0][0])



NWORKERS = 12
training = []
validation = []
half_t = [(0, 49), (50, len(dataset_train) - 1)]
half_v = [(0, 31),(32, len(dataset_validate) - 1)]
for i in range(2):
    training.append(DataLoader(dataset=dataset_train,
                    batch_size=1,
                    drop_last=False,
                    num_workers=NWORKERS,
                    sampler= SequentialSampler(torch.arange(half_t[i][0], half_t[i][1]))))


for i in range(2):
    validation.append(DataLoader(dataset=dataset_validate,
                    batch_size=1,
                    drop_last=False,
                    num_workers=NWORKERS,
                    sampler= SequentialSampler(torch.arange(half_v[i][0], half_v[i][1]))))
    


def main(acc_hist, loss_hist):
    # K clients
    # C The ratio of the clients we will include
    # E local Epochs
    # B batch size
    # r Global epochs
    x, y = 1, 2
    K, C, E, B, r = 3, 1, 10, 1, 5
    input_dim = 28
    lr = 0.08
    options = {'K': K, 'C': C, 'E': E, 'B': B, 'r': r,
                "training_data": training, "validation_data": validation,
                  'input_dim': input_dim, 'lr': lr}
    
    scaffold = Scaffold(options)
    model, acc_hist, loss_hist = scaffold.server()
    scaffold.global_test(model)
#     return acc_hist, loss_hist
    print("loss_hist: ", loss_hist)
    print("acc_list: ", acc_hist)
    
    plot_acc_loss("SCAFFOLD iid", loss_hist, acc_hist)
    
if __name__ == '__main__':
    acc_hist = []
    loss_hist = []
    main(acc_hist, loss_hist)
    
