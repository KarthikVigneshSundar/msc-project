import json
import os
import re

import torch
from pykeen.datasets import FB15k237, FB15k, WN18

from models import TransE
from train_and_evaluate import TransETrain, Evaluation
from transe_boost import TransEBoost, TransEBoost2


class LoadData:
    def __init__(self, folder_list):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.dataset_name = None
        self.num_entity = None
        self.num_relation = None
        self.folder_list = folder_list
        # check if the necessary folders exist
        exist_dir = [dir_name for dir_name in os.listdir('.') if os.path.isdir(dir_name) if
                     dir_name in self.folder_list]
        if len(exist_dir) != len(self.folder_list):
            dir_to_create = [dir_name for dir_name in self.folder_list if dir_name not in exist_dir]
            for direc in dir_to_create:
                os.mkdir(direc)

    def resume_exp(self, folder):
        print('Experiment List: ')
        for exp in [dir_name for dir_name in os.listdir(folder + '/') if os.path.isdir(dir_name)]:
            print(exp)
        select_exp_num = input('Choose a Exp. number from list: ')
        exp_dir_name = folder + '/' + 'exp_' + str(select_exp_num)
        with open(exp_dir_name + 'meta_data.json', 'r') as json_file:
            meta_data = json.load(json_file)
            json_file.close()
        dataset_name = meta_data['global']['dataset name']
        if dataset_name == 'FB15k':
            automatic_input = '1'
        elif dataset_name == 'FB15k237':
            automatic_input = '2'
        elif dataset_name == 'WN18':
            automatic_input = '3'
        self.get_dataset(automatic_input=automatic_input)
        return meta_data, exp_dir_name

    def load(self, folder):
        choose_exp = input('Run a new experiment (y), or continue with an existing one (n) ?: ')
        if choose_exp == 'y':
            meta_data, exp_dir_name = self.create_exp(folder=folder)
            return meta_data, exp_dir_name, self.train_dataset, self.val_dataset, self.test_dataset, self.num_entity, self.num_relation
        elif choose_exp == 'n':
            meta_data, exp_dir_name = self.resume_exp(folder=folder)
            return meta_data, exp_dir_name, self.train_dataset, self.val_dataset, self.test_dataset, self.num_entity, self.num_relation

    def get_latest_exp(self, folder):
        exist_dir = [dir_name for dir_name in os.listdir(folder + '/') if os.path.isdir(dir_name)]
        if len(exist_dir) != 0:
            file_numbers = [int(re.findall('\d+', dir_name)[0]) for dir_name in exist_dir]
            if len(file_numbers) != 0:
                latest_exp_num = max(file_numbers)
            else:
                return 0
        else:
            return 0
        return latest_exp_num

    def create_exp(self, folder):
        new_exp_num = self.get_latest_exp(folder) + 1
        exp_dir_name = folder + '/' + 'exp_' + new_exp_num
        os.mkdir(exp_dir_name)
        self.get_dataset()
        meta_data = {'global': {'device': input('Device: '),
                                'seed': int(input('Seed: ')),
                                'dataset name': self.dataset_name,
                                'num of entity': self.num_entity,
                                'num of relation': self.num_relation,
                                'emb dim': int(input('Embedding Dimension: ')),
                                'gamma': float(input('Gamma: ')),
                                'lr': float(input('Learning rate: ')),
                                'l2': float(input('L2 Weight Decay: ')),
                                'batch size': int(input('Batch Size: ')),
                                'latest epoch': -1},
                     'local': {}
                     }
        with open(exp_dir_name + '/' + 'meta_data.json', 'w+') as json_file:
            json.dump(meta_data, json_file, indent=4)
            json_file.close()
        return meta_data, exp_dir_name

    def get_dataset(self, automatic_input=None):
        if automatic_input == None:
            select_dataset = input('''1)FB15K\n2)FB15K237\n3)WN18\n4)Exit\nEnter 1,2,3 or 4:''')
        else:
            select_dataset = automatic_input
        if select_dataset == '1':
            dataset = FB15k()
            self.dataset_name = 'FB15k'
        elif select_dataset == '2':
            dataset = FB15k237()
            self.dataset_name = 'FB15k237'
        elif select_dataset == '3':
            dataset = WN18()
            self.dataset_name = 'WN18'
        elif select_dataset == '4':
            return
        else:
            return
        # assign the values
        self.train_dataset = dataset.training.mapped_triples
        self.val_dataset = dataset.validation.mapped_triples
        self.test_dataset = dataset.testing.mapped_triples
        self.num_entity = dataset.num_entities.real
        self.num_relation = dataset.num_relations.real
        # print their shapes
        print('Training dataset size: ', self.train_dataset.shape)
        print('Validation dataset size: ', self.val_dataset.shape)
        print('Testing dataset size: ', self.test_dataset.shape)
        print('Number of Entities: ', self.num_entity)
        print('Number of relations: ', self.num_relation)
        print('Dataset Name: ', self.dataset_name)
        # return the values
        # return train_dataset, val_dataset, test_dataset, num_entity, num_relation, dataset_name


# set transe model paratmeters:
device = 'cuda'
emb_dim = 50
gamma = 1

# set transe optimizer parameters:
lr = 0.01
weight_decay = 0

# transe training parameters:
batch_size = 36
epoch = 50

# set torch seeds:
seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# TransEBoost parameters:
end_epoch = 100


def main():
    load_data = LoadData()
    train_dataset, val_dataset, test_dataset, num_entity, num_relation, dataset_name = load_data.get_dataset()

    select_train_model = input('''
                               \n1) Train a basic TransE model? 
                               \n2) Evaluate a model? 
                               \n3) Training a self-training type 1 model?
                               \n4) Training a self-training type 2 model?
                               \n5) Exit 
                               \n Enter (1, 2, 3, 4, 5): ''')

    if select_train_model == '1':
        # Create the transe model:
        print('Creating a new TransE model: ')
        transe_model = TransE(device=device,
                              num_entity=num_entity,
                              num_relation=num_relation,
                              emb_dim=emb_dim,
                              gamma=gamma,
                              seed=seed)
        print('Done!!')
        # Create the optimizer:
        optimizer = torch.optim.SGD(transe_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)
        # Train the model:
        transe_model_train = TransETrain(train_dataset=train_dataset,
                                         batch_size=batch_size,
                                         num_entity=num_entity,
                                         model=transe_model,
                                         device=device,
                                         optimizer=optimizer,
                                         epoch=epoch,
                                         seed=seed)
        # train TransE model:
        transe_model_train.train_transe()
        # save the model:
        save_original_transe(model=transe_model,
                             folder='transe_neworg_models',
                             num_entity=num_entity,
                             num_relation=num_relation,
                             dataset_name=dataset_name,
                             mr=0,
                             mrr=0,
                             hits_at_10=0)

    elif select_train_model == '2':
        transe_model = load_original_transe(num_entity=num_entity, num_relation=num_relation)
        if transe_model == -1:
            return -1
        # Create the optimizer:
        optimizer = torch.optim.SGD(transe_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)

        # setup to train and evaluate the model:
        transe_model_train = TransETrain(train_dataset=train_dataset,
                                         batch_size=batch_size,
                                         num_entity=num_entity,
                                         model=transe_model,
                                         device=device,
                                         optimizer=optimizer,
                                         epoch=epoch,
                                         seed=seed)
        # train TransE model:
        transe_model_train.train_transe()
        # save model
        save_original_transe(model=transe_model,
                             folder='transe_conorg_models',
                             num_entity=num_entity,
                             num_relation=num_relation,
                             dataset_name=dataset_name,
                             mr=0,
                             mrr=0,
                             hits_at_10=0)

    elif select_train_model == '3':
        select_eva_model = input('''Select model to be evaluated:\n
                                  1) original model\n
                                  2) continue training original model\n
                                  3) boosted model\n
                                  4) boosted 2 model\n
                                  Enter (1,2,3):''')
        select_eva_model_num = input('''Enter the model number: ''')
        if select_eva_model == '1':
            evaluate_model = 'transe_neworg_models/transe_org_model_'
        elif select_eva_model == '2':
            evaluate_model = 'transe_conorg_models/transe_org_model_'
        elif select_eva_model == '3':
            evaluate_model = 'transe_boosted_models/transe_boost_model_'
        elif select_eva_model == '4':
            evaluate_model = 'transe_boosted2_models/transe_boost2_model_'
        evaluate_model += str(select_eva_model_num) + '.pt'
        print('Loading a TransE model from disk...')
        try:
            transe_model = torch.load(evaluate_model)
        except:
            print('Error in loading model. Aborting.')
            return -1
        print('Done!\nModel being evaluated: ', evaluate_model)
        # specify the values for evaluation:
        eva_obj = Evaluation(data=val_dataset,
                             model=transe_model,
                             num_entity=num_entity,
                             device=device)
        # Evaluate TransE model:
        mr, mrr, hits_at_10 = eva_obj.evaluate_model()
        # save evaluation results:
        save_evaluation_res(mr=mr,
                            mrr=mrr,
                            hits_at_10=hits_at_10,
                            dataset_name=dataset_name,
                            evaluate_model=evaluate_model)

    elif select_train_model == '4':
        transe_model, start_epoch = restore_boosted_model(num_entity=num_entity, num_relation=num_relation)
        if transe_model == -1:
            return -1
        if start_epoch >= end_epoch:
            print('Already trained till end epoch condition')
            return 0
        # Create the optimizer:
        optimizer = torch.optim.SGD(transe_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)
        # setup to train the model:
        boost_obj = TransEBoost(start_epoch=start_epoch,
                                end_epoch=end_epoch,
                                train_data=train_dataset,
                                val_data=val_dataset,
                                seed=seed,
                                device=device,
                                batch_size=batch_size,
                                model=transe_model,
                                optimizer=optimizer,
                                num_entity=num_entity)
        # save the meta data:
        save_boosted_meta(dataset_name=dataset_name,
                          num_entity=num_entity,
                          num_relation=num_relation,
                          start_epoch=start_epoch)
        # train the model:
        boost_obj.train()

    elif select_train_model == '5':
        transe_model = torch.load('transe_conorg_models/transe_org_model_1.pt')
        # Create the optimizer:
        optimizer = torch.optim.SGD(transe_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)
        # setup the transe boost 2
        boost2_obj = TransEBoost2(pre_model=transe_model,
                                  cur_model=transe_model,
                                  train_data=train_dataset,
                                  start_epoch=1,
                                  end_epoch=10,
                                  seed=seed,
                                  device=device,
                                  batch_size=batch_size,
                                  start_model=1,
                                  end_model=5,
                                  optimizer=optimizer,
                                  num_entity=num_entity)
        # train:
        boost2_obj.train()

    elif select_train_model == '6':
        return 0


def restore_boosted_model(num_entity, num_relation):
    file_list = os.listdir('transe_boosted_models/')
    if len(file_list) == 0:
        print('Creating a new TransE model: ')
        transe_model = TransE(device=device,
                              num_entity=num_entity,
                              num_relation=num_relation,
                              emb_dim=emb_dim,
                              gamma=gamma,
                              seed=seed)
        print('Done!!')
        start_epoch = 1
    else:
        print('Loading last trained model...')
        file_numbers = [int(re.findall('\d+', i)[0]) for i in file_list]
        model_num = max(file_numbers)
        try:
            transe_model = torch.load('transe_boosted_models/transe_boost_model_' + str(model_num) + '.pt')
        except:
            print('Error encountered while loading model. Aborting.')
            return -1, -1
        start_epoch = model_num + 1
        print('Done!')
    return transe_model, start_epoch


def save_original_transe(model, folder, num_entity, num_relation, dataset_name, mr, mrr, hits_at_10):
    file_list = os.listdir(folder + '/')
    meta_data = {'Device': device,
                 'Seed': seed,
                 'Dataset Name': dataset_name,
                 'Number of Entities': num_entity,
                 'Number of Relations': num_relation,
                 'Embedding Dimension': emb_dim,
                 'Gamma': gamma,
                 'Learning Rate': lr,
                 'L2': weight_decay,
                 'Model Number': 0,
                 'Epochs': epoch,
                 'Batch Size': batch_size,
                 'MR': float(mr),
                 'MRR': float(mrr),
                 'Hits@10': float(hits_at_10)}
    if len(file_list) == 0:
        meta_data['Model Number'] = 1
    else:
        file_numbers = [int(re.findall('\d+', i)[0]) for i in file_list]
        model_num = max(file_numbers)
        model_num += 1
        meta_data['Model Number'] = model_num
    torch.save(model, folder + '/transe_org_model_' + str(meta_data['Model Number']) + '.pt')
    with open(folder + '/meta_data_' + str(meta_data['Model Number']) + '.json', 'w') as json_file:
        json.dump(meta_data, json_file, indent=4)
        json_file.close()
    print('Saving done !!')


def load_original_transe(num_entity, num_relation):
    file_list = os.listdir('transe_conorg_models/')
    if len(file_list) == 0:
        print('No existing model found')
        print('Creating a new TransE model: ')
        transe_model = TransE(device=device,
                              num_entity=num_entity,
                              num_relation=num_relation,
                              emb_dim=emb_dim,
                              gamma=gamma,
                              seed=seed)
        print('Done!!')
    else:
        file_numbers = [int(re.findall('\d+', i)[0]) for i in file_list]
        model_num = max(file_numbers)
        try:
            transe_model = torch.load('transe_conorg_models/transe_org_model_' + str(model_num) + '.pt')
        except:
            print('Error encountered while loading model. Aborting.')
            return -1
    return transe_model


def save_evaluation_res(mr, mrr, hits_at_10, dataset_name, evaluate_model):
    meta_data = {'Device': device,
                 'Seed': seed,
                 'Dataset Name': dataset_name,
                 'Model Name': evaluate_model.replace('/', '-'),
                 'MR': float(mr),
                 'MRR': float(mrr),
                 'Hits@10': float(hits_at_10)}
    with open('evaluation_results/meta_data_' + meta_data['Model Name'][:-3] + '.json', 'w') as json_file:
        json.dump(meta_data, json_file, indent=4)
        json_file.close()
    print('Saving done !!')


def save_boosted_meta(dataset_name, num_entity, num_relation, start_epoch):
    meta_data = {'Device': device,
                 'Seed': seed,
                 'Dataset Name': dataset_name,
                 'Number of Entities': num_entity,
                 'Number of Relations': num_relation,
                 'Embedding Dimension': emb_dim,
                 'Gamma': gamma,
                 'Learning Rate': lr,
                 'L2': weight_decay,
                 'Start Epoch': start_epoch,
                 'Batch Size': batch_size}
    with open('transe_boosted_models/meta_data' + str(meta_data['Start Epoch']) + '.json', 'w') as json_file:
        json.dump(meta_data, json_file, indent=4)
        json_file.close()
    print('Saving Done!!')


if __name__ == '__main__':
    main()
