import sys
import torch
from tqdm import tqdm
import numpy as np

from data_loader import DataLoader
from graftnet import GraftNet
from util import use_cuda, save_model, load_model, get_config, load_dict, cal_accuracy
from util import load_documents, index_document_entities, output_pred_dist

import pickle

def train(cfg):
    print("training ...")

    # prepare data
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])

    train_documents = load_documents(cfg['data_folder'] + cfg['train_documents'])
    train_document_entity_indices, train_document_texts = index_document_entities(train_documents, word2id, entity2id, cfg['max_document_word'])
    train_data = DataLoader(cfg['data_folder'] + cfg['train_data'], train_documents, train_document_entity_indices, train_document_texts, word2id, relation2id, entity2id, cfg['max_query_word'], cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation']) 
    
    if cfg['dev_documents'] != cfg['train_documents']:
        valid_documents = load_documents(cfg['data_folder'] + cfg['dev_documents'])
        valid_document_entity_indices, valid_document_texts = index_document_entities(valid_documents, word2id, entity2id, cfg['max_document_word'])
    else:
        valid_documents = train_documents
        valid_document_entity_indices, valid_document_texts = train_document_entity_indices, train_document_texts
    valid_data = DataLoader(cfg['data_folder'] + cfg['dev_data'], valid_documents, valid_document_entity_indices, valid_document_texts, word2id, relation2id, entity2id, cfg['max_query_word'], cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'])
    
    if cfg['test_documents'] != cfg['dev_documents']:
        test_documents = load_documents(cfg['data_folder'] + cfg['test_documents'])
        test_document_entity_indices, test_document_texts = index_document_entities(test_documents, word2id, entity2id, cfg['max_document_word'])
    else:
        test_documents = valid_documents
        test_document_entity_indices, test_document_texts = valid_document_entity_indices, valid_document_texts
    test_data = DataLoader(cfg['data_folder'] + cfg['test_data'], test_documents, test_document_entity_indices, test_document_texts, word2id, relation2id, entity2id, cfg['max_query_word'], cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'])

    # create model & set parameters
    my_model = get_model(cfg, train_data.num_kb_relation, len(entity2id), len(word2id))
    trainable_parameters = [p for p in my_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg['learning_rate'])

    best_dev_acc = 0.0
    for epoch in range(cfg['num_epoch']):
        try:
            print('epoch', epoch)
            train_data.reset_batches(is_sequential = cfg['is_debug'])
            # Train
            my_model.train()
            train_loss, train_acc, train_max_acc = [], [], []
            for iteration in tqdm(range(train_data.num_data // cfg['train_batch_size'])):
                batch, sample_ids = train_data.get_batch(iteration, cfg['train_batch_size'], cfg['fact_dropout'])

                print('---------')
                print(type(batch))
                print(len(batch))
                sample_ids = sample_ids.tolist()
                print(type(sample_ids), len(sample_ids))

                doc_score_original_train = []
                for i in sample_ids:
                    doc_score_original_train.append([x['retrieval_score'] for x in train_data.data[i]['passages']])

                print('--------------')
                loss, _, _ = my_model(batch, doc_score_original_train)
                # pred = pred.data.cpu().numpy()
                
                # acc, max_acc = cal_accuracy(pred, batch[-1])
                train_loss.append(loss.data[0])
                # train_acc.append(acc)
                # train_max_acc.append(max_acc)
                # back propogate
                my_model.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(my_model.parameters(), cfg['gradient_clip'])
                optimizer.step()
            print('avg_training_loss', sum(train_loss) / len(train_loss))
            # print('max_training_acc', sum(train_max_acc) / len(train_max_acc))
            # print('avg_training_acc', sum(train_acc) / len(train_acc))

            print("validating ...")
            eval_acc = inference(my_model, valid_data, entity2id, cfg)
            if eval_acc > best_dev_acc and cfg['to_save_model']:
                print("saving model to", cfg['save_model_file'])
                torch.save(my_model.state_dict(), cfg['save_model_file'])
                best_dev_acc = eval_acc

        except KeyboardInterrupt:
            break

    # Test set evaluation
    print("evaluating on test")
    print('loading model from ...', cfg['save_model_file'])
    my_model.load_state_dict(torch.load(cfg['save_model_file']))
    test_acc = inference(my_model, test_data, entity2id, cfg, log_info=True)

    return test_acc


def inference(my_model, valid_data, entity2id, cfg, log_info=False):
    # Evaluation
    my_model.eval()
    eval_loss, eval_acc, eval_max_acc = [], [], []
    id2entity = {idx: entity for entity, idx in entity2id.items()}
    valid_data.reset_batches(is_sequential = True)

    testndev_batch_size = cfg['testndev_batch_size']

    if log_info:
        f_pred = open(cfg['pred_file'], 'w')
    for iteration in tqdm(range(valid_data.num_data // testndev_batch_size)):
        batch, sample_ids = valid_data.get_batch(iteration, testndev_batch_size, fact_dropout=0.0)

        sample_ids = sample_ids.tolist()

        # print(type(sample_ids), sample_ids)
        doc_score_original = []
        for i in sample_ids:
            doc_score_original.append([x['retrieval_score'] for x in valid_data.data[i]['passages']])
        # print(doc_score_original)
        # doc_descending_original.sort(key = lambda x:x['retrieval_score'], reverse=True)
        
        loss, pred, pred_dist = my_model(batch, doc_score_original)

        # doc_score = doc_score.detach().cpu().numpy()
        # doc_indexes_score_wise = valid_data.rel_document_ids
        # doc_index_score = sorted(set(zip(doc_indexes_score_wise[0], doc_score[0])), key=lambda x:x[1], reverse=True)
        # doc_id_sorted = [x for x,_ in doc_index_score]

        # print(doc_ranking_original)
        # print(doc_id_sorted)

        pred = pred.data.cpu().numpy()
        acc, max_acc = cal_accuracy(pred, batch[-1])

        if log_info: 
            output_pred_dist(pred_dist, batch[-1], id2entity, iteration * testndev_batch_size, valid_data, f_pred)
        
        eval_loss.append(loss.data[0])
        eval_acc.append(acc)
        eval_max_acc.append(max_acc)

    if eval_loss:
        print('avg_loss', sum(eval_loss) / len(eval_loss))
    if eval_max_acc:
        print('max_acc', sum(eval_max_acc) / len(eval_max_acc))
    if eval_acc:
        print('avg_acc', sum(eval_acc) / len(eval_acc))

    return sum(eval_acc) / len(eval_acc)

def test(cfg):
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])

    test_documents = load_documents(cfg['data_folder'] + cfg['test_documents'])
    test_document_entity_indices, test_document_texts = index_document_entities(test_documents, word2id, entity2id, cfg['max_document_word'])
    test_data = DataLoader(cfg['data_folder'] + cfg['test_data'], test_documents, test_document_entity_indices, test_document_texts, word2id, relation2id, entity2id, cfg['max_query_word'], cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'])

    # print(test_data)
    # with open(r'/home/tarun/ResearchWork/GraftNet/manually_created_files/read_test_try1_check', 'wb') as file1:
    #     pickle.dump(test_data, file1)
    
    # print("TESTDATA", type(test_data))
    # print(test_data.num_data, test_data.num_kb_relation)
    
    my_model = get_model(cfg, test_data.num_kb_relation, len(entity2id), len(word2id))

    # print(my_model)
    # print(type(my_model))

    # with open(r'/home/tarun/ResearchWork/GraftNet/manually_created_files/', 'wb') as file1:
    #     pickle.dump(test_data, file1)

    test_acc = inference(my_model, test_data, entity2id, cfg, log_info=True)
    return test_acc


def get_model(cfg, num_kb_relation, num_entities, num_vocab):
    word_emb_file = None if cfg['word_emb_file'] is None else cfg['data_folder'] + cfg['word_emb_file']
    entity_emb_file = None if cfg['entity_emb_file'] is None else cfg['data_folder'] + cfg['entity_emb_file']
    entity_kge_file = None if cfg['entity_kge_file'] is None else cfg['data_folder'] + cfg['entity_kge_file']
    relation_emb_file = None if cfg['relation_emb_file'] is None else cfg['data_folder'] + cfg['relation_emb_file']
    relation_kge_file = None if cfg['relation_kge_file'] is None else cfg['data_folder'] + cfg['relation_kge_file']
    
    my_model = use_cuda(GraftNet(word_emb_file, entity_emb_file, entity_kge_file, relation_emb_file, relation_kge_file, cfg['num_layer'], num_kb_relation, num_entities, num_vocab, cfg['entity_dim'], cfg['word_dim'], cfg['kge_dim'], cfg['pagerank_lambda'], cfg['fact_scale'], cfg['lstm_dropout'], cfg['linear_dropout'], cfg['use_kb'], cfg['use_doc'])) 

    if cfg['load_model_file'] is not None:
        print('loading model from', cfg['load_model_file'])
        pretrained_model_states = torch.load(cfg['load_model_file'])
        if word_emb_file is not None:
            del pretrained_model_states['word_embedding.weight']
        if entity_emb_file is not None:
            del pretrained_model_states['entity_embedding.weight']
        my_model.load_state_dict(pretrained_model_states, strict=False)
    
    return my_model

if __name__ == "__main__":
    config_file = sys.argv[2]
    CFG = get_config(config_file)
    if '--train' == sys.argv[1]:
        train(CFG)
    elif '--test' == sys.argv[1]:
        test(CFG)
    else:
        assert False, "--train or --test?"

