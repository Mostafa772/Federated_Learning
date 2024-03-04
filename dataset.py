from requirements import *

class Dataset:
    def __init__(self, path):
        self.path = path
        if self.path[-1] != '/':
            self.path += '/'
        if self.path[-2] == 'n':
            self.df = pd.read_csv(self.path + 'fragments-t.csv')
        else:
            self.df = pd.read_csv(self.path + 'fragments-v.csv')
        self.NFFF = 200

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        sid = str(self.df.iloc[item]['segment_id']) + ".mat" 
        # print(type(sid))
        target = self.df.iloc[item]['category_id']
        try:
            mat_file = mat73.loadmat(self.path+'{}'.format(sid))
        except:
            mat_file = sio.loadmat(self.path+'{}'.format(sid))  
        # print(mat_file)
        data = mat_file["data"]

        _,_, data = signal.spectrogram(data,fs=5000,nperseg=256,noverlap=128,nfft=1024)

        data = data[:self.NFFF,:]
        data = stats.zscore(data,axis=1)
        data = np.expand_dims(data,axis=0)
        data = np.float32(data)
        return data,target

    def split_reviewer(self,reviewer_id):
        train = copy.deepcopy(self)
        valid = copy.deepcopy(self)

        idx = self.df['reviewer_id']!=reviewer_id

        train.df = train.df[idx].reset_index(drop=True)
        valid.df = valid.df[np.logical_not(idx)].reset_index(drop=True)
        return train,valid

    def split_random(self,N_valid):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        train = copy.deepcopy(self)
        valid = copy.deepcopy(self)

        train.df = train.df.iloc[N_valid:].reset_index(drop=True)
        valid.df = valid.df.iloc[:N_valid].reset_index(drop=True)
        return train,valid

    def integrity_check(self):
        # iterate through dataset and check if all the files might be correctly loaded
        try:
            for i in tqdm(range(len(self))):
                x = self.__getitem__(i)
        except Exception as exc:
            raise exc

    def remove_powerline_noise_class(self):
        self.df = self.df[self.df['category_id']!=0]
        self.df['category_id'] = self.df['category_id'] - 1
        self.df = self.df.reset_index(drop=True)
        return self


if __name__ == "__main__":
    dataset_train = Dataset('./new_data/train/').integrity_check()
    dataset_validate = Dataset('./new_data/validate/').integrity_check()








# cleint_2 = DataLoader(dataset=dataset_train,
#                    batch_size=1,
#                    drop_last=False,
#                    num_workers=NWORKERS,
#                    sampler= SequentialSampler(torch.arange(50, len(dataset_train)-1)))

# cleint_v1 = DataLoader(dataset=dataset_validate,
#                    batch_size=1,
#                    sampler= SequentialSampler(torch.arange(0, 31)),
#                    drop_last=False,
#                    num_workers=NWORKERS)
# cleint_v1 = DataLoader(dataset=dataset_validate,
#                    batch_size=1,
#                    sampler= SequentialSampler(torch.arange(32, len(dataset_validate)-1)),
#                    drop_last=False,
#                    num_workers=NWORKERS)

# cleint_1, cleint_2= torch.utils.data.random_split(
#         TRAIN, [48, 47], generator=torch.Generator().manual_seed(1))
# print(next(iter(cleint_1)))
# loader_1 = DataLoader(cleint_1, batch_size=len(cleint_1))
# dataiter = iter(loader_1)
# d, t = next(dataiter)
# print(d, t)
# print(len(loader_1))

# print(cleint_1.__getitem__(0))
