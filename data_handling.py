import json
import os
import re
import torch
from pykeen.datasets import FB15k237, FB15k, WN18

from models import TransE


class LoadMetaDataHandling:
    def __init__(self, folder_list, dataset_num_map):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.dataset_name = None
        self.num_entity = None
        self.num_relation = None
        self.folder_list = folder_list
        self.dataset_num_map = dataset_num_map
        # check if the necessary folders exist
        exist_dir = [dir_name for dir_name in os.listdir('./') if os.path.isdir('./' + dir_name) if
                     dir_name in self.folder_list]
        if len(exist_dir) != len(self.folder_list):
            dir_to_create = [dir_name for dir_name in self.folder_list if dir_name not in exist_dir]
            for direc in dir_to_create:
                os.mkdir(direc)

    def dataset_name_to_number(self, dataset_name):
        return self.dataset_num_map[dataset_name]

    def resume_exp(self, folder):
        print('Experiment List: ')
        for exp in [dir_name for dir_name in os.listdir('./' + folder + '/') if
                    os.path.isdir('./' + folder + '/' + dir_name)]:
            print(exp)
        select_exp_num = input('Choose a Exp. number from list: ')
        exp_dir_name = folder + '/' + 'exp_' + str(select_exp_num)
        with open(exp_dir_name + '/' + 'meta_data.json', 'r') as json_file:
            meta_data = json.load(json_file)
            json_file.close()
        automatic_input = self.dataset_name_to_number(dataset_name=meta_data['global']['dataset name'])
        self.get_dataset(automatic_input=automatic_input)
        transe_model = torch.load(exp_dir_name + '/' + 'transe_' + str(meta_data['global']['latest epoch']) + '.pt')
        return meta_data, exp_dir_name, transe_model

    def load(self, folder):
        choose_exp = input('Run a new experiment (y), or continue with an existing one (n) ?: ')
        if choose_exp == 'y':
            meta_data, exp_dir_name, transe_model = self.create_exp(folder=folder)
        else:
            meta_data, exp_dir_name, transe_model = self.resume_exp(folder=folder)
        torch.manual_seed(meta_data['global']['seed'])
        torch.cuda.manual_seed_all(meta_data['global']['seed'])
        return meta_data, exp_dir_name, transe_model, self.train_dataset, self.val_dataset, self.test_dataset

    def get_latest_exp(self, folder):
        exist_dir = [dir_name for dir_name in os.listdir('./' + folder + '/') if
                     os.path.isdir('./' + folder + '/' + dir_name)]
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
        exp_dir_name = folder + '/' + 'exp_' + str(new_exp_num)
        os.mkdir(exp_dir_name)
        self.get_dataset()
        meta_data = {'global': {'device': input('Device: '),
                                'seed': int(input('Seed: ')),
                                'dataset name': self.dataset_name,
                                'num entity': self.num_entity,
                                'num relation': self.num_relation,
                                'emb dim': int(input('Embedding Dimension: ')),
                                'gamma': float(input('Gamma: ')),
                                'lr': float(input('Learning rate: ')),
                                'l2': float(input('L2 Weight Decay: ')),
                                'batch size': int(input('Batch Size: ')),
                                'latest epoch': 0,
                                'total epoch': int(input('total number of epochs: ')),
                                'start model': None,
                                'description': input('Enter a brief description about this experiment: ')
                                },
                     'local': {}
                     }
        # Choose the source of transe model to start training from:
        transe_model_select = input('Create a new model (y) or use a existing model (n): ')
        if transe_model_select == 'y':
            # create a new transe model:
            print('Creating a new TransE model: ')
            transe_model = {'cur_model': TransE(device=meta_data['global']['device'],
                                                num_entity=meta_data['global']['num entity'],
                                                num_relation=meta_data['global']['num relation'],
                                                emb_dim=meta_data['global']['emb dim'],
                                                gamma=meta_data['global']['gamma'],
                                                seed=meta_data['global']['seed'])
                            }
            print('Done!!')
            # Set the start model choice in meta-data:
            meta_data['global']['start model'] = 'new TransE model'
        else:
            transe_model, model_path = self.get_model()
            # Set the start model choice in meta-data:
            meta_data['global']['start model'] = model_path
        # Save the meta-data to disk:
        with open(exp_dir_name + '/' + 'meta_data.json', 'w+') as json_file:
            json.dump(meta_data, json_file, indent=4)
            json_file.close()
        return meta_data, exp_dir_name, transe_model

    def get_model(self):
        print('Select a folder: ')
        for index, folder in enumerate(self.folder_list):
            print(str(index + 1) + ') ' + folder)
        folder_choice = input('Enter a number: ')
        exp_choice = input('Enter a exp. number: ')
        model_choice = input('Enter a model number: ')
        folder = self.folder_list[int(folder_choice) - 1]
        model_path = folder + '/' + 'exp_' + exp_choice + '/' + 'transe_' + model_choice + '.pt'
        transe_model = torch.load(model_path)
        return transe_model, model_path

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

    def get_data(self):
        return self.train_dataset, self.val_dataset, self.test_dataset

    def meta_data_add_field(self, exp_dir_name, **kwargs):
        with open(exp_dir_name + '/' + 'meta_data.json', 'r+') as json_file:
            meta_data = json.load(json_file)
            for key, value in kwargs.items():
                meta_data['global'][str(key)] = value
            json_file.seek(0)
            json.dump(meta_data, json_file, indent=4)
            json_file.truncate()
            json_file.close()
            return meta_data

class Draw:
    def __init__(self):
        pass