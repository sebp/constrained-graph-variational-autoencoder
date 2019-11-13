#!/usr/bin/env/python
"""
Usage:
    CGVAE.py [options]

Options:
    -h --help                Show this screen
    --dataset NAME           Dataset name: zinc, qm9, cep
    --config-file FILE       Hyperparameter configuration file path (in JSON format)
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format)
    --log_dir NAME           log dir name
    --data_dir NAME          data dir name
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components
"""
from typing import Sequence, Any
from docopt import docopt
from collections import defaultdict, deque
import numpy as np
import tensorflow as tf
import sys, traceback
import json
import os
from GGNN_core import ChemModel
import utils
from utils import *
import pickle
import random
from numpy import linalg as LA
from rdkit import Chem
from copy import deepcopy
from rdkit.Chem import QED
import os
import time
from data_augmentation import *

'''
Comments provide the expected tensor shapes where helpful.

Key to symbols in comments:
---------------------------
[...]:  a tensor
; ; :   a list
b:      batch size
e:      number of edege types (3)
es:     maximum number of BFS transitions in this batch
v:      number of vertices per graph in this batch
h:      GNN hidden size
'''

class DenseGGNNChemModel(ChemModel):
    def __init__(self, args):
        super().__init__(args)

        self.make_minibatch_iterator = self._make_minibatch_iterator_from_file

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
                        'task_sample_ratios': {},
                        'use_edge_bias': True,       # whether use edge bias in gnn

                        'clamp_gradient_norm': 1.0,
                        'out_layer_dropout_keep_prob': 1.0,

                        'tie_fwd_bkwd': True,
                        'task_ids': [0],             # id of property prediction

                        'random_seed': 0,            # fixed for reproducibility 
                       
                        'batch_size': 8 if dataset=='zinc' or dataset=='cep' else 64,              
                        "qed_trade_off_lambda": 10,
                        'num_epochs': 3 if dataset=='zinc' or dataset=='cep' else 10,
                        'epoch_to_generate': 3 if dataset=='zinc' or dataset=='cep' else 10,
                        'number_of_generation': 50000,
                        'maximum_distance': 10,
                        "use_argmax_generation": False,    # use random sampling or argmax during generation
                        'residual_connection_on': False,    # whether residual connection is on
                        'residual_connections': {          # For iteration i, specify list of layers whose output is added as an input
                                2: [0],
                                4: [0, 2],
                                6: [0, 2, 4],
                                8: [0, 2, 4, 6],
                                10: [0, 2, 4, 6, 8],
                                12: [0, 2, 4, 6, 8, 10],
                                14: [0, 2, 4, 6, 8, 10, 12],
                            },
                        'num_timesteps': 7,           # gnn propagation step
                        'hidden_size': 100,        
                        "kl_trade_off_lambda": 0.3,    # kl tradeoff
                        'learning_rate': 0.001, 
                        'graph_state_dropout_keep_prob': 1,    
                        "compensate_num": 1,           # how many atoms to be added during generation

                        'train_file': 'data/molecules_train_%s.json' % dataset,
                        'valid_file': 'data/molecules_valid_%s.json' % dataset,

                        'try_different_starting': False,
                        "num_different_starting": 1,

                        'generation': False,        # only for generation
                        'use_graph': True,          # use gnn
                        "label_one_hot": False,     # one hot label or not
                        "multi_bfs_path": True,    # whether sample several BFS paths for each molecule
                        "bfs_path_count": 6,
                        "path_random_order": True, # False: canonical order, True: random order
                        "sample_transition": False, # whether use transition sampling
                        'edge_weight_dropout_keep_prob': 1,
                        'check_overlap_edge': False,
                        "truncate_distance": 10,
                        })

        return params

    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']        
        expanded_h_dim=self.params['hidden_size']+self.params['hidden_size'] + 1 # 1 for focus bit
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None, name='edge_weight_dropout_keep_prob')
        self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32,
                                                                          [None, None, self.params['hidden_size']],
                                                                          name='node_features')  # padded node symbols
        # mask out invalid node
        self.placeholders['node_mask'] = tf.placeholder(tf.float32, [None, None], name='node_mask') # [b x v]
        self.placeholders['num_vertices'] = tf.placeholder(tf.int32, ())
        # adj for encoder
        self.placeholders['adjacency_matrix'] = tf.placeholder(tf.float32,
                                                    [None, self.num_edge_types, None, None], name="adjacency_matrix")     # [b, e, v, v]
        # labels for node symbol prediction
        self.placeholders['node_symbols'] = tf.placeholder(tf.float32, [None, None, self.params['num_symbols']]) # [b, v, edge_type]
        # node symbols used to enhance latent representations
        self.placeholders['latent_node_symbols'] = tf.placeholder(tf.float32, 
                                                      [None, None, self.params['hidden_size']], name='latent_node_symbol') # [b, v, h]
        # mask out cross entropies in decoder
        self.placeholders['iteration_mask']=tf.placeholder(tf.float32, [None, None]) # [b, es]
        # adj matrices used in decoder
        self.placeholders['incre_adj_mat']=tf.placeholder(tf.float32, [None, None, self.num_edge_types, None, None], name='incre_adj_mat') # [b, es, e, v, v]
        # distance 
        self.placeholders['distance_to_others']=tf.placeholder(tf.int32, [None, None, None], name='distance_to_others') # [b, es,v]
        # maximum iteration number of this batch
        self.placeholders['max_iteration_num']=tf.placeholder(tf.int32, [], name='max_iteration_num') # number
        # node number in focus at each iteration step
        self.placeholders['node_sequence']=tf.placeholder(tf.float32, [None, None, None], name='node_sequence') # [b, es, v]
        # mask out invalid edge types at each iteration step 
        self.placeholders['edge_type_masks']=tf.placeholder(tf.float32, [None, None, self.num_edge_types, None], name='edge_type_masks') # [b, es, e, v]
        # ground truth edge type labels at each iteration step 
        self.placeholders['edge_type_labels']=tf.placeholder(tf.float32, [None, None, self.num_edge_types, None], name='edge_type_labels') # [b, es, e, v]
        # mask out invalid edge at each iteration step 
        self.placeholders['edge_masks']=tf.placeholder(tf.float32, [None, None, None], name='edge_masks') # [b, es, v]
        # ground truth edge labels at each iteration step 
        self.placeholders['edge_labels']=tf.placeholder(tf.float32, [None, None, None], name='edge_labels') # [b, es, v]        
        # ground truth labels for whether it stops at each iteration step
        self.placeholders['local_stop']=tf.placeholder(tf.float32, [None, None], name='local_stop') # [b, es]
        # z_prior sampled from standard normal distribution
        self.placeholders['z_prior']=tf.placeholder(tf.float32, [None, None, self.params['hidden_size']], name='z_prior') # the prior of z sampled from normal distribution
        # put in front of kl latent loss
        self.placeholders['kl_trade_off_lambda']=tf.placeholder(tf.float32, [], name='kl_trade_off_lambda') # number

        # weights for encoder and decoder GNN. 
        if self.params["residual_connection_on"]:
            # weights for encoder and decoder GNN. Different weights for each iteration
            for scope in ['_encoder', '_decoder']:
                if scope == '_encoder':
                    new_h_dim=h_dim
                else:
                    new_h_dim=expanded_h_dim
                for iter_idx in range(self.params['num_timesteps']):
                    with tf.variable_scope("gru_scope"+scope+str(iter_idx), reuse=False):
                        self.weights['edge_weights'+scope+str(iter_idx)] = tf.Variable(
                            glorot_init([self.num_edge_types, new_h_dim, new_h_dim]), name='edge_weights')
                        if self.params['use_edge_bias']:
                            self.weights['edge_biases'+scope+str(iter_idx)] = tf.Variable(
                                np.zeros([self.num_edge_types, 1, new_h_dim]).astype(np.float32), name='edge_biases')
                
                        cell = tf.contrib.rnn.GRUCell(new_h_dim)
                        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                        state_keep_prob=self.placeholders['graph_state_keep_prob'])
                        self.weights['node_gru'+scope+str(iter_idx)] = cell
        else:
            for scope in ['_encoder', '_decoder']:
                if scope == '_encoder':
                    new_h_dim=h_dim
                else:
                    new_h_dim=expanded_h_dim
                self.weights['edge_weights'+scope] = tf.Variable(
                    glorot_init([self.num_edge_types, new_h_dim, new_h_dim]), name='edge_weights'+scope)
                if self.params['use_edge_bias']:
                    self.weights['edge_biases'+scope] = tf.Variable(
                        np.zeros([self.num_edge_types, 1, new_h_dim]).astype(np.float32), name='edge_biases'+scope)
                with tf.variable_scope("gru_scope"+scope):
                    cell = tf.contrib.rnn.GRUCell(new_h_dim)
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                         state_keep_prob=self.placeholders['graph_state_keep_prob'])
                    self.weights['node_gru'+scope] = cell

        # weights for calculating mean and variance
        self.weights['mean_weights'] = tf.Variable(glorot_init([h_dim, h_dim]), name='mean_weights')
        self.weights['mean_biases'] = tf.Variable(np.zeros([1, h_dim]).astype(np.float32), name='mean_biases')
        self.weights['variance_weights'] = tf.Variable(glorot_init([h_dim, h_dim]), name='variance_weights')
        self.weights['variance_biases'] = tf.Variable(np.zeros([1, h_dim]).astype(np.float32), name='variance_biases')

        # The weights for generating nodel symbol logits    
        self.weights['node_symbol_weights'] = tf.Variable(glorot_init([h_dim, self.params['num_symbols']]),
            name='node_symbol_weights')
        self.weights['node_symbol_biases'] = tf.Variable(np.zeros([1, self.params['num_symbols']]).astype(np.float32),
            name='node_symbol_biases')

        feature_dimension=5*expanded_h_dim
        # record the total number of features
        self.params["feature_dimension"] = 5
        # weights for generating edge type logits

        self.weights['edge_type'] = tf.Variable(glorot_init([feature_dimension, feature_dimension]),
            name='kernel_edge_type')
        self.weights['edge_type_biases'] = tf.Variable(np.zeros([1, feature_dimension]).astype(np.float32),
            name='edge_type_biases')
        self.weights['edge_type_output'] = tf.Variable(glorot_init([feature_dimension, self.num_edge_types]),
            name='kernel_edge_type_output')
        # weights for generating edge logits
        self.weights['edge_iteration'] = tf.Variable(glorot_init([feature_dimension, feature_dimension]),
            name='kernel_edge_iteration')
        self.weights['edge_iteration_biases'] = tf.Variable(np.zeros([1, feature_dimension]).astype(np.float32),
            name='kernel_edge_iteration_biases')
        self.weights['edge_iteration_output'] = tf.Variable(glorot_init([feature_dimension, 1]),
            name='kernel_edge_iteration_output')
        # Weights for the stop node
        self.weights["stop_node"] = tf.Variable(glorot_init([1, expanded_h_dim]), name='kernel_stop_node')
        # Weight for distance embedding
        self.weights['distance_embedding'] = tf.Variable(glorot_init([self.params['maximum_distance'],
            expanded_h_dim]), name='kernel_distance_embedding')
        # use node embeddings
        self.weights["node_embedding"]= tf.Variable(glorot_init([self.params["num_symbols"], h_dim]),
            name='kernel_node_embedding')
        
        # graph state mask
        self.ops['graph_state_mask']= tf.expand_dims(self.placeholders['node_mask'], 2)

    # transform one hot vector to dense embedding vectors
    def get_node_embedding_state(self, one_hot_state):
        node_nums=tf.argmax(one_hot_state, axis=2)
        return tf.nn.embedding_lookup(self.weights["node_embedding"], node_nums) * self.ops['graph_state_mask']

    def compute_final_node_representations_with_residual(self, h, adj, scope_name): # scope_name: _encoder or _decoder
        # h: initial representation, adj: adjacency matrix, different GNN parameters for encoder and decoder
        v = self.placeholders['num_vertices']
        # _decoder uses a larger latent space because concat of symbol and latent representation
        if scope_name=="_decoder":
            h_dim = self.params['hidden_size'] + self.params['hidden_size'] + 1
        else:
            h_dim = self.params['hidden_size']
        h = tf.reshape(h, [-1, h_dim]) # [b*v, h]
        # record all hidden states at each iteration
        all_hidden_states=[h]
        for iter_idx in range(self.params['num_timesteps']):
            with tf.variable_scope("gru_scope"+scope_name+str(iter_idx), reuse=None) as g_scope:
                for edge_type in range(self.num_edge_types):
                    # the message passed from this vertice to other vertices
                    m = tf.matmul(h, self.weights['edge_weights'+scope_name+str(iter_idx)][edge_type])  # [b*v, h]
                    if self.params['use_edge_bias']:
                        m += self.weights['edge_biases'+scope_name+str(iter_idx)][edge_type]            # [b, v, h]
                    m = tf.reshape(m, [-1, v, h_dim])                                                   # [b, v, h]
                    # collect the messages from other vertices to each vertice
                    if edge_type == 0:
                        acts = tf.matmul(adj[edge_type], m)
                    else:
                        acts += tf.matmul(adj[edge_type], m)
                # all messages collected for each node
                acts = tf.reshape(acts, [-1, h_dim])                                                    # [b*v, h]
                # add residual connection here
                layer_residual_connections = self.params['residual_connections'].get(iter_idx)
                if layer_residual_connections is None:
                    layer_residual_states = []
                else:
                    layer_residual_states = [all_hidden_states[residual_layer_idx]
                                             for residual_layer_idx in layer_residual_connections]
                # concat current hidden states with residual states
                acts= tf.concat([acts] + layer_residual_states, axis=1)                                 # [b, (1+num residual connection)* h]

                # feed msg inputs and hidden states to GRU
                h = self.weights['node_gru'+scope_name+str(iter_idx)](acts, h)[1]                       # [b*v, h]
                # record the new hidden states
                all_hidden_states.append(h)
        last_h = tf.reshape(all_hidden_states[-1], [-1, v, h_dim])
        return last_h

    def compute_final_node_representations_without_residual(self, h, adj, edge_weights, edge_biases, node_gru, gru_scope_name): 
    # h: initial representation, adj: adjacency matrix, different GNN parameters for encoder and decoder
        v = self.placeholders['num_vertices']
        if gru_scope_name=="gru_scope_decoder":
            h_dim = self.params['hidden_size'] + self.params['hidden_size'] + 1
        else:
            h_dim = self.params['hidden_size']
        h = tf.reshape(h, [-1, h_dim])

        with tf.variable_scope(gru_scope_name) as scope:
            for i in range(self.params['num_timesteps']):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                for edge_type in range(self.num_edge_types):
                    m = tf.matmul(h, tf.nn.dropout(edge_weights[edge_type],
                               keep_prob=self.placeholders['edge_weight_dropout_keep_prob']))           # [b*v, h]
                    if self.params['use_edge_bias']:
                        m += edge_biases[edge_type]                                                     # [b, v, h]
                    m = tf.reshape(m, [-1, v, h_dim])                                                   # [b, v, h]
                    if edge_type == 0:
                        acts = tf.matmul(adj[edge_type], m)
                    else:
                        acts += tf.matmul(adj[edge_type], m)
                acts = tf.reshape(acts, [-1, h_dim])                                                    # [b*v, h]
                h = node_gru(acts, h)[1]                                                                # [b*v, h]
            last_h = tf.reshape(h, [-1, v, h_dim])
        return last_h

    def compute_mean_and_logvariance(self):
        h_dim = self.params['hidden_size']
        reshped_last_h=tf.reshape(self.ops['final_node_representations'], [-1, h_dim])
        mean=tf.matmul(reshped_last_h, self.weights['mean_weights']) + self.weights['mean_biases']
        logvariance=tf.matmul(reshped_last_h, self.weights['variance_weights']) + self.weights['variance_biases']
        return mean, logvariance
        
    def sample_with_mean_and_logvariance(self):
        v = self.placeholders['num_vertices']
        h_dim = self.params['hidden_size']
        # Sample from normal distribution
        z_prior = tf.reshape(self.placeholders['z_prior'], [-1, h_dim])
        # Train: sample from u, Sigma. Generation: sample from 0,1
        z_sampled = tf.cond(self.placeholders['is_generative'], lambda: z_prior, # standard normal 
                    lambda: tf.add(self.ops['mean'], tf.multiply(tf.sqrt(tf.exp(self.ops['logvariance'])), z_prior))) # non-standard normal
        # filter
        z_sampled = tf.reshape(z_sampled, [-1, v, h_dim]) * self.ops['graph_state_mask'] 
        return z_sampled

    def fully_connected(self, input, hidden_weight, hidden_bias, output_weight):
        if input.get_shape().ndims > 2:
            matmul = lambda x, y: tf.tensordot(x, y, [[2], [0]])
        else:
            matmul = tf.matmul
        output=tf.nn.relu(matmul(input, hidden_weight) + hidden_bias)
        output=matmul(output, output_weight)
        return output

    def generate_cross_entropy(self, idx, cross_entropy_losses, edge_predictions, edge_type_predictions):
        v = self.placeholders['num_vertices']
        h_dim = self.params['hidden_size']
        num_symbols = self.params['num_symbols']
        batch_size = tf.shape(self.placeholders['initial_node_representation'])[0]
        # Use latent representation as decoder GNN'input 
        filtered_z_sampled = self.ops["initial_repre_for_decoder"]                                      # [b, v, h+h]
        # data needed in this iteration
        incre_adj_mat = self.placeholders['incre_adj_mat'][:,idx,:,:, :]                                # [b, e, v, v]
        distance_to_others = self.placeholders['distance_to_others'][:, idx, :]                         # [b,v]
        node_sequence = self.placeholders['node_sequence'][:, idx, :] # [b, v]
        node_sequence = tf.expand_dims(node_sequence, axis=2) # [b,v,1]
        edge_type_masks = self.placeholders['edge_type_masks'][:, idx, :, :] # [b, e, v]
        # make invalid locations to be very small before using softmax function
        edge_type_masks = edge_type_masks * LARGE_NUMBER - LARGE_NUMBER
        edge_type_labels = self.placeholders['edge_type_labels'][:, idx, :, :] # [b, e, v]
        edge_masks=self.placeholders['edge_masks'][:, idx, :] # [b, v]
        # make invalid locations to be very small before using softmax function
        edge_masks = edge_masks * LARGE_NUMBER - LARGE_NUMBER
        edge_labels = self.placeholders['edge_labels'][:, idx, :] # [b, v]  
        local_stop = self.placeholders['local_stop'][:, idx] # [b]        
        # concat the hidden states with the node in focus
        filtered_z_sampled = tf.concat([filtered_z_sampled, node_sequence], axis=2) # [b, v, h + h + 1]
        # Decoder GNN
        if self.params["use_graph"]:
            if self.params["residual_connection_on"]:
                new_filtered_z_sampled = self.compute_final_node_representations_with_residual(filtered_z_sampled,   
                                                    tf.transpose(incre_adj_mat, [1, 0, 2, 3]), 
                                                    "_decoder") # [b, v, h + h]
            else:
                new_filtered_z_sampled = self.compute_final_node_representations_without_residual(filtered_z_sampled,   
                                                tf.transpose(incre_adj_mat, [1, 0, 2, 3]), 
                                                self.weights['edge_weights_decoder'], 
                                                self.weights['edge_biases_decoder'], 
                                                self.weights['node_gru_decoder'], "gru_scope_decoder") # [b, v, h + h]
        else:
            new_filtered_z_sampled = filtered_z_sampled
        # Filter nonexist nodes
        new_filtered_z_sampled=new_filtered_z_sampled * self.ops['graph_state_mask']
        # Take out the node in focus
        node_in_focus = tf.reduce_sum(node_sequence * new_filtered_z_sampled, axis=1)# [b, h + h]
        # edge pair representation
        edge_repr=tf.concat(\
            [tf.tile(tf.expand_dims(node_in_focus, 1), [1,v,1]), new_filtered_z_sampled], axis=2) # [b, v, 2*(h+h)]            
        #combine edge repre with local and global repr
        local_graph_repr_before_expansion = tf.reduce_sum(new_filtered_z_sampled, axis=1) /  \
                                            tf.reduce_sum(self.placeholders['node_mask'], axis=1, keep_dims=True) # [b, h + h]
        local_graph_repr = tf.expand_dims(local_graph_repr_before_expansion, 1)        
        local_graph_repr = tf.tile(local_graph_repr, [1,v,1])  # [b, v, h+h]        
        global_graph_repr_before_expansion = tf.reduce_sum(filtered_z_sampled, axis=1) / \
                                            tf.reduce_sum(self.placeholders['node_mask'], axis=1, keep_dims=True) 
        global_graph_repr = tf.expand_dims(global_graph_repr_before_expansion, 1)
        global_graph_repr = tf.tile(global_graph_repr, [1,v,1]) # [b, v, h+h]
        # distance representation
        distance_repr = tf.nn.embedding_lookup(self.weights['distance_embedding'], distance_to_others) # [b, v, h+h]

        # concat and reshape.
        combined_edge_repr = tf.concat([edge_repr, local_graph_repr,
                                       global_graph_repr, distance_repr], axis=2)
        # Calculate edge logits
        edge_logits=self.fully_connected(combined_edge_repr, self.weights['edge_iteration'],
                                        self.weights['edge_iteration_biases'], self.weights['edge_iteration_output'])
        edge_logits=tf.reshape(edge_logits, [-1, v]) # [b, v]
        # filter invalid terms
        edge_logits=edge_logits + edge_masks
        # Calculate whether it will stop at this step
        # prepare the data
        expanded_stop_node = tf.tile(self.weights['stop_node'], [batch_size, 1]) # [b, h + h]
        distance_to_stop_node = tf.nn.embedding_lookup(self.weights['distance_embedding'], tf.tile([0], [batch_size]))     # [b, h + h]
         
        combined_stop_node_repr = tf.concat([node_in_focus, expanded_stop_node, local_graph_repr_before_expansion, 
                                     global_graph_repr_before_expansion, distance_to_stop_node], axis=1) # [b, 6 * (h + h)]
        # logits for stop node                                    
        stop_logits = self.fully_connected(combined_stop_node_repr, 
                            self.weights['edge_iteration'], self.weights['edge_iteration_biases'],
                            self.weights['edge_iteration_output']) #[b, 1]
        edge_logits = tf.concat([edge_logits, stop_logits], axis=1) # [b, v + 1]

        # Calculate edge type logits
        edge_type_logits = self.fully_connected(combined_edge_repr,
                            self.weights['edge_type'], self.weights['edge_type_biases'],
                            self.weights['edge_type_output']) # [b, v, e]
        edge_type_logits = tf.transpose(edge_type_logits, [0, 2, 1]) # [b, e, v]

        # filter invalid items
        edge_type_logits = edge_type_logits + edge_type_masks # [b, e, v]
        # softmax over edge type axis
        edge_type_probs = tf.nn.softmax(edge_type_logits, 1) # [b, e, v]

        # edge labels
        edge_labels = tf.concat([edge_labels,tf.expand_dims(local_stop, 1)], axis=1) # [b, v + 1]                
        # softmax for edge
        edge_loss =- tf.reduce_sum(tf.log(tf.nn.softmax(edge_logits) + SMALL_NUMBER) * edge_labels, axis=1)
        # softmax for edge type 
        edge_type_loss =- edge_type_labels * tf.log(edge_type_probs + SMALL_NUMBER) # [b, e, v]
        edge_type_loss = tf.reduce_sum(edge_type_loss, axis=[1, 2]) # [b]
        # total loss
        iteration_loss = edge_loss + edge_type_loss
        cross_entropy_losses = cross_entropy_losses.write(idx, iteration_loss)
        edge_predictions = edge_predictions.write(idx, tf.nn.softmax(edge_logits))
        edge_type_predictions = edge_type_predictions.write(idx, edge_type_probs)
        return (idx+1, cross_entropy_losses, edge_predictions, edge_type_predictions)

    def construct_logit_matrices(self):
        v = self.placeholders['num_vertices']
        batch_size=tf.shape(self.placeholders['initial_node_representation'])[0]
        h_dim = self.params['hidden_size']

        # Initial state: embedding
        latent_node_state= self.get_node_embedding_state(self.placeholders["latent_node_symbols"])
        # concat z_sampled with node symbols
        filtered_z_sampled = tf.concat([self.ops['z_sampled'],
                                       latent_node_state], axis=2) # [b, v, h + h]
        self.ops["initial_repre_for_decoder"] = filtered_z_sampled
        # The tensor array used to collect the cross entropy losses at each step
        cross_entropy_losses = tf.TensorArray(dtype=tf.float32, size=self.placeholders['max_iteration_num'])
        edge_predictions= tf.TensorArray(dtype=tf.float32, size=self.placeholders['max_iteration_num'])
        edge_type_predictions = tf.TensorArray(dtype=tf.float32, size=self.placeholders['max_iteration_num'])
        idx_final, cross_entropy_losses_final, edge_predictions_final,edge_type_predictions_final=\
            tf.while_loop(lambda idx, cross_entropy_losses,edge_predictions,edge_type_predictions: idx < self.placeholders['max_iteration_num'],
            self.generate_cross_entropy,
            (tf.constant(0), cross_entropy_losses,edge_predictions,edge_type_predictions,))
        
        # record the predictions for generation
        self.ops['edge_predictions'] = edge_predictions_final.read(0)
        self.ops['edge_type_predictions'] = edge_type_predictions_final.read(0)

        # final cross entropy losses
        cross_entropy_losses_final = cross_entropy_losses_final.stack()
        self.ops['cross_entropy_losses'] = tf.transpose(cross_entropy_losses_final, [1,0]) # [b, es]

        # Logits for node symbols
        self.ops['node_symbol_logits']=tf.reshape(tf.matmul(tf.reshape(self.ops['z_sampled'],[-1, h_dim]), self.weights['node_symbol_weights']) + 
                                                  self.weights['node_symbol_biases'], [-1, v, self.params['num_symbols']])

    def construct_loss(self):
        v = self.placeholders['num_vertices']
        h_dim = self.params['hidden_size']
        kl_trade_off_lambda =self.placeholders['kl_trade_off_lambda']
        # Edge loss
        self.ops["edge_loss"] = tf.reduce_sum(self.ops['cross_entropy_losses'] * self.placeholders['iteration_mask'], axis=1)
        # KL loss 
        kl_loss = 1 + self.ops['logvariance'] - tf.square(self.ops['mean']) - tf.exp(self.ops['logvariance'])
        kl_loss = tf.reshape(kl_loss, [-1, v, h_dim]) * self.ops['graph_state_mask'] 
        self.ops['kl_loss'] = -0.5 * tf.reduce_sum(kl_loss, [1,2])
        # Node symbol loss
        self.ops['node_symbol_prob'] = tf.nn.softmax(self.ops['node_symbol_logits'])
        self.ops['node_symbol_loss'] = -tf.reduce_sum(tf.log(self.ops['node_symbol_prob'] + SMALL_NUMBER) * 
                                                      self.placeholders['node_symbols'], axis=[1,2])
        # Add in the loss for calculating QED
        self.ops['mean_edge_loss'] = tf.reduce_mean(self.ops["edge_loss"]) # record the mean edge loss
        self.ops['mean_node_symbol_loss'] = tf.reduce_mean(self.ops["node_symbol_loss"])
        self.ops['mean_kl_loss'] = tf.reduce_mean(kl_trade_off_lambda *self.ops['kl_loss'])
        total_loss = tf.reduce_mean(self.ops["edge_loss"] + self.ops['node_symbol_loss'] + \
                              kl_trade_off_lambda *self.ops['kl_loss'])
        for nam in ('mean_edge_loss', 'mean_node_symbol_loss', 'mean_kl_loss'):
            tf.summary.scalar(nam, self.ops[nam])
        tf.summary.scalar('loss', total_loss)
        return total_loss

    def gated_regression(self, last_h, regression_gate, regression_transform, hidden_size, projection_weight, projection_bias, v, mask):
        # last_h: [b x v x h]
        last_h = tf.reshape(last_h, [-1, hidden_size])   # [b*v, h]    
        # linear projection on last_h
        last_h = tf.nn.relu(tf.matmul(last_h, projection_weight)+projection_bias) # [b*v, h]  
        # same as last_h
        gate_input = last_h
        # linear projection and combine                                       
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * tf.nn.tanh(regression_transform(last_h)) # [b*v, 1]
        gated_outputs = tf.reshape(gated_outputs, [-1, v])                  # [b, v]
        masked_gated_outputs = gated_outputs * mask                           # [b x v]
        output = tf.reduce_sum(masked_gated_outputs, axis = 1)                                                # [b]
        output=tf.sigmoid(output)
        return output

    def calculate_incremental_results(self, raw_data, bucket_sizes, file_name):
        incremental_results=[]
        # copy the raw_data if more than 1 BFS path is added
        new_raw_data=[]
        for idx, d in enumerate(raw_data):
            # Use canonical order or random order here. canonical order starts from index 0. random order starts from random nodes
            if not self.params["path_random_order"]:
                # Use several different starting index if using multi BFS path
                if self.params["multi_bfs_path"]:
                    list_of_starting_idx= list(range(self.params["bfs_path_count"]))
                else:
                    list_of_starting_idx=[0] # the index 0
            else:
                # get the node length for this molecule
                node_length=len(d["node_features"])
                if self.params["multi_bfs_path"]:
                    list_of_starting_idx= np.random.choice(node_length, self.params["bfs_path_count"], replace=True) #randomly choose several
                else:
                    list_of_starting_idx= [random.choice(list(range(node_length)))] # randomly choose one
            for list_idx, starting_idx in enumerate(list_of_starting_idx):
                # choose a bucket
                chosen_bucket_idx = np.argmax(bucket_sizes > max([v for e in d['graph']
                                                                    for v in [e[0], e[2]]]))
                chosen_bucket_size = bucket_sizes[chosen_bucket_idx]

                # Calculate incremental results without master node
                nodes_no_master, edges_no_master = to_graph(d['smiles'], self.params["dataset"])                
                incremental_adj_mat,distance_to_others,node_sequence,edge_type_masks,edge_type_labels,local_stop, edge_masks, edge_labels, overlapped_edge_features=\
                construct_incremental_graph(dataset, edges_no_master, chosen_bucket_size, 
                                            len(nodes_no_master), nodes_no_master, self.params, initial_idx=starting_idx)
                if self.params["sample_transition"] and list_idx > 0:
                    incremental_results[-1]=[x+y for x, y in zip(incremental_results[-1], [incremental_adj_mat,distance_to_others,
                                       node_sequence,edge_type_masks,edge_type_labels,local_stop, edge_masks, edge_labels, overlapped_edge_features])]
                else:
                    incremental_results.append([incremental_adj_mat, distance_to_others, node_sequence, edge_type_masks, 
                                               edge_type_labels, local_stop, edge_masks, edge_labels, overlapped_edge_features])
                    # copy the raw_data here 
                    new_raw_data.append(d)
                if idx % 50 == 0:
                    print('finish calculating %d incremental matrices' % idx, end="\r")
        return incremental_results, new_raw_data

    # ----- Data preprocessing and chunking into minibatches:
    def process_raw_graphs(self, raw_data, is_training_data, file_name, bucket_sizes=None):
        if bucket_sizes is None:
            bucket_sizes = dataset_info(self.params["dataset"])["bucket_sizes"]
        incremental_results, raw_data=self.calculate_incremental_results(raw_data, bucket_sizes, file_name)
        bucketed = defaultdict(list)
        x_dim = len(raw_data[0]["node_features"][0])

        for d, (incremental_adj_mat,distance_to_others,node_sequence,edge_type_masks,edge_type_labels,local_stop, edge_masks, edge_labels, overlapped_edge_features)\
                            in zip(raw_data, incremental_results):
            # choose a bucket
            chosen_bucket_idx = np.argmax(bucket_sizes > max([v for e in d['graph']
                                                                for v in [e[0], e[2]]]))
            chosen_bucket_size = bucket_sizes[chosen_bucket_idx]            
            # total number of nodes in this data point
            n_active_nodes = len(d["node_features"])
            bucketed[chosen_bucket_idx].append({
                'adj_mat': graph_to_adj_mat(d['graph'], chosen_bucket_size, self.num_edge_types, self.params['tie_fwd_bkwd']),
                'incre_adj_mat': incremental_adj_mat,
                'distance_to_others': distance_to_others,
                'overlapped_edge_features': overlapped_edge_features,
                'node_sequence': node_sequence,
                'edge_type_masks': edge_type_masks,
                'edge_type_labels': edge_type_labels,
                'edge_masks': edge_masks,
                'edge_labels': edge_labels,
                'local_stop': local_stop,
                'number_iteration': len(local_stop),
                'init': d["node_features"] + [[0 for _ in range(x_dim)] for __ in
                                              range(chosen_bucket_size - n_active_nodes)],
                'labels': [d["targets"][task_id][0] for task_id in self.params['task_ids']],
                'mask': [1. for _ in range(n_active_nodes) ] + [0. for _ in range(chosen_bucket_size - n_active_nodes)]
            })

        if is_training_data:
            for (bucket_idx, bucket) in bucketed.items():
                np.random.shuffle(bucket)
                for task_id in self.params['task_ids']:
                    task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
                    if task_sample_ratio is not None:
                        ex_to_sample = int(len(bucket) * task_sample_ratio)
                        for ex_id in range(ex_to_sample, len(bucket)):
                            bucket[ex_id]['labels'][task_id] = None

        bucket_at_step = [[bucket_idx for _ in range(len(bucket_data) // self.params['batch_size'])]
                          for bucket_idx, bucket_data in bucketed.items()]
        bucket_at_step = [x for y in bucket_at_step for x in y]

        return (bucketed, bucket_sizes, bucket_at_step)

    def pad_annotations(self, annotations):
        return np.pad(annotations,
                       pad_width=[[0, 0], [0, 0], [0, self.params['hidden_size'] - self.params["num_symbols"]]],
                       mode='constant')

    def make_batch(self, elements, maximum_vertice_num):
        # get maximum number of iterations in this batch. used to control while_loop
        max_iteration_num=-1
        for d in elements:
            max_iteration_num=max(d['number_iteration'], max_iteration_num)
        batch_data = {'adj_mat': [], 'init': [], 'labels': [], 'edge_type_masks':[], 'edge_type_labels':[], 'edge_masks':[],                 
                'edge_labels':[],'node_mask': [], 'task_masks': [], 'node_sequence':[],
                'iteration_mask': [], 'local_stop': [], 'incre_adj_mat': [], 'distance_to_others': [], 
                'max_iteration_num': max_iteration_num, 'overlapped_edge_features': []}
        for d in elements: 
            # sparse to dense for saving memory           
            incre_adj_mat = incre_adj_mat_to_dense(d['incre_adj_mat'], self.num_edge_types, maximum_vertice_num)
            distance_to_others = distance_to_others_dense(d['distance_to_others'], maximum_vertice_num)
            overlapped_edge_features = overlapped_edge_features_to_dense(d['overlapped_edge_features'], maximum_vertice_num)
            node_sequence = node_sequence_to_dense(d['node_sequence'],maximum_vertice_num)
            edge_type_masks = edge_type_masks_to_dense(d['edge_type_masks'], maximum_vertice_num,self.num_edge_types)
            edge_type_labels = edge_type_labels_to_dense(d['edge_type_labels'], maximum_vertice_num,self.num_edge_types)
            edge_masks = edge_masks_to_dense(d['edge_masks'], maximum_vertice_num)
            edge_labels = edge_labels_to_dense(d['edge_labels'], maximum_vertice_num)

            batch_data['adj_mat'].append(d['adj_mat'])
            batch_data['init'].append(d['init'])
            batch_data['node_mask'].append(d['mask'])

            batch_data['incre_adj_mat'].append(incre_adj_mat +
                [np.zeros((self.num_edge_types, maximum_vertice_num,maximum_vertice_num)) 
                            for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['distance_to_others'].append(distance_to_others + 
                [np.zeros((maximum_vertice_num)) 
                            for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['overlapped_edge_features'].append(overlapped_edge_features + 
                [np.zeros((maximum_vertice_num)) 
                            for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['node_sequence'].append(node_sequence + 
                [np.zeros((maximum_vertice_num)) 
                            for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['edge_type_masks'].append(edge_type_masks + 
                [np.zeros((self.num_edge_types, maximum_vertice_num)) 
                            for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['edge_masks'].append(edge_masks + 
                [np.zeros((maximum_vertice_num)) 
                            for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['edge_type_labels'].append(edge_type_labels + 
                [np.zeros((self.num_edge_types, maximum_vertice_num)) 
                            for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['edge_labels'].append(edge_labels + 
                [np.zeros((maximum_vertice_num)) 
                            for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['iteration_mask'].append([1 for _ in range(d['number_iteration'])]+
                                     [0 for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['local_stop'].append([int(s) for s in d["local_stop"]]+ 
                                     [0 for _ in range(max_iteration_num-d['number_iteration'])])

            target_task_values = []
            target_task_mask = []
            for target_val in d['labels']:
                if target_val is None:  # This is one of the examples we didn't sample...
                    target_task_values.append(0.)
                    target_task_mask.append(0.)
                else:
                    target_task_values.append(target_val)
                    target_task_mask.append(1.)
            batch_data['labels'].append(target_task_values)
            batch_data['task_masks'].append(target_task_mask)

        return batch_data

    def get_dynamic_feed_dict(self, elements, latent_node_symbol, incre_adj_mat, num_vertices, 
                    distance_to_others, node_sequence, edge_type_masks, edge_masks, random_normal_states):
        if incre_adj_mat is None:
            incre_adj_mat=np.zeros((1, 1, self.num_edge_types, 1, 1))
            distance_to_others=np.zeros((1,1,1))
            node_sequence=np.zeros((1,1,1))
            edge_type_masks=np.zeros((1,1,self.num_edge_types,1))
            edge_masks=np.zeros((1,1,1))
            latent_node_symbol=np.zeros((1,1,self.params["num_symbols"]))
        return {
                self.placeholders['z_prior']: random_normal_states, # [1, v, h]
                self.placeholders['incre_adj_mat']: incre_adj_mat, # [1, 1, e, v, v]
                self.placeholders['num_vertices']: num_vertices,     # v

                self.placeholders['initial_node_representation']: \
                            self.pad_annotations([elements['init']]),
                self.placeholders['node_symbols']: [elements['init']],
                self.placeholders['latent_node_symbols']: self.pad_annotations(latent_node_symbol),
                self.placeholders['adjacency_matrix']: [elements['adj_mat']],
                self.placeholders['node_mask']: [elements['mask']], 

                self.placeholders['graph_state_keep_prob']: 1, 
                self.placeholders['edge_weight_dropout_keep_prob']: 1,               
                self.placeholders['iteration_mask']: [[1]],
                self.placeholders['is_generative']: True, 
                self.placeholders['out_layer_dropout_keep_prob'] : 1.0, 
                self.placeholders['distance_to_others'] : distance_to_others, # [1, 1,v]
                self.placeholders['max_iteration_num']: 1,
                self.placeholders['node_sequence']: node_sequence, #[1, 1, v]
                self.placeholders['edge_type_masks']: edge_type_masks, #[1, 1, e, v]
                self.placeholders['edge_masks']: edge_masks, # [1, 1, v]
            }

    def get_node_symbol(self, batch_feed_dict):  
        fetch_list = [self.ops['node_symbol_prob']]
        result = self.sess.run(fetch_list, feed_dict=batch_feed_dict)
        return result[0]

    def node_symbol_one_hot(self, sampled_node_symbol, real_n_vertices, max_n_vertices):
        one_hot_representations=[]
        for idx in range(max_n_vertices):
            representation = [0] * self.params["num_symbols"]
            if idx < real_n_vertices:
                atom_type=sampled_node_symbol[idx]
                representation[atom_type]=1
            one_hot_representations.append(representation)
        return one_hot_representations

    def search_and_generate_molecule(self, initial_idx, valences, 
                             sampled_node_symbol, real_n_vertices, random_normal_states,
                             elements, max_n_vertices):
        # New molecule
        new_mol = Chem.MolFromSmiles('')
        new_mol = Chem.rdchem.RWMol(new_mol)
        # Add atoms
        add_atoms(new_mol, sampled_node_symbol, self.params["dataset"])
        # Breadth first search over the molecule
        queue=deque([initial_idx])
        # color 0: have not found 1: in the queue 2: searched already
        color = [0] * max_n_vertices
        color[initial_idx] = 1
        # Empty adj list at the beginning
        incre_adj_list=defaultdict(list)
        # record the log probabilities at each step
        total_log_prob=0
        while len(queue) > 0:
            node_in_focus = queue.popleft()
            # iterate until the stop node is selected 
            while True:
                # Prepare data for one iteration based on the graph state
                edge_type_mask_sparse, edge_mask_sparse = generate_mask(valences, incre_adj_list, color, real_n_vertices, node_in_focus, self.params["check_overlap_edge"], new_mol)
                edge_type_mask = edge_type_masks_to_dense([edge_type_mask_sparse], max_n_vertices, self.num_edge_types) # [1, e, v]
                edge_mask = edge_masks_to_dense([edge_mask_sparse],max_n_vertices) # [1, v]
                node_sequence = node_sequence_to_dense([node_in_focus], max_n_vertices) # [1, v]
                distance_to_others_sparse = bfs_distance(node_in_focus, incre_adj_list)
                distance_to_others = distance_to_others_dense([distance_to_others_sparse],max_n_vertices) # [1, v]
                incre_adj_mat = incre_adj_mat_to_dense([incre_adj_list], 
                    self.num_edge_types, max_n_vertices) # [1, e, v, v]
                sampled_node_symbol_one_hot = self.node_symbol_one_hot(sampled_node_symbol, real_n_vertices, max_n_vertices)

                # get feed_dict
                feed_dict=self.get_dynamic_feed_dict(elements, [sampled_node_symbol_one_hot],
                            [incre_adj_mat], max_n_vertices, [distance_to_others],
                            [node_sequence], [edge_type_mask], [edge_mask], random_normal_states)

                # fetch nn predictions
                fetch_list = [self.ops['edge_predictions'], self.ops['edge_type_predictions']]
                edge_probs, edge_type_probs = self.sess.run(fetch_list, feed_dict=feed_dict)
                # select an edge
                if not self.params["use_argmax_generation"]:
                    neighbor=np.random.choice(np.arange(max_n_vertices+1), p=edge_probs[0])
                else:
                    neighbor=np.argmax(edge_probs[0])
                # update log prob
                total_log_prob+=np.log(edge_probs[0][neighbor]+SMALL_NUMBER)
                # stop it if stop node is picked
                if neighbor == max_n_vertices:
                    break                    
                # or choose an edge type
                if not self.params["use_argmax_generation"]:
                    bond=np.random.choice(np.arange(self.num_edge_types),p=edge_type_probs[0, :, neighbor])
                else:
                    bond=np.argmax(edge_type_probs[0, :, neighbor])
                # update log prob
                total_log_prob+=np.log(edge_type_probs[0, :, neighbor][bond]+SMALL_NUMBER)
                #update valences
                valences[node_in_focus] -= (bond+1)
                valences[neighbor] -= (bond+1)
                #add the bond
                new_mol.AddBond(int(node_in_focus), int(neighbor), number_to_bond[bond])
                # add the edge to increment adj list
                incre_adj_list[node_in_focus].append((neighbor, bond))
                incre_adj_list[neighbor].append((node_in_focus, bond))
                # Explore neighbor nodes
                if color[neighbor]==0:
                    queue.append(neighbor)
                    color[neighbor]=1                
            color[node_in_focus]=2    # explored
        # Remove unconnected node     
        remove_extra_nodes(new_mol)
        new_mol=Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))
        return new_mol, total_log_prob

    def gradient_ascent(self, random_normal_states, derivative_z_sampled):        
        return random_normal_states + self.params['prior_learning_rate'] * derivative_z_sampled

    # optimization in latent space. generate one molecule for each optimization step
    def optimization_over_prior(self, random_normal_states, num_vertices, generated_all_similes, elements, count):
        # record how many optimization steps are taken
        step=0
        # generate a new molecule
        self.generate_graph_with_state(random_normal_states, num_vertices, generated_all_similes, elements, step, count)
        return random_normal_states


    def generate_graph_with_state(self, random_normal_states, num_vertices,
                                  generated_all_similes, elements, step, count):
        # Get back node symbol predictions
        # Prepare dict
        node_symbol_batch_feed_dict=self.get_dynamic_feed_dict(elements, None, None,
                                     num_vertices, None, None, None, None, random_normal_states)
        # Get predicted node probs
        predicted_node_symbol_prob=self.get_node_symbol(node_symbol_batch_feed_dict)
        # Node numbers for each graph
        real_length=get_graph_length([elements['mask']])[0] # [valid_node_number] 
        # Sample node symbols
        sampled_node_symbol=sample_node_symbol(predicted_node_symbol_prob, [real_length], self.params["dataset"])[0] # [v]        
        # Maximum valences for each node
        valences=get_initial_valence(sampled_node_symbol, self.params["dataset"]) # [v]
        # randomly pick the starting point or use zero 
        if not self.params["path_random_order"]:
            # Try different starting points
            if self.params["try_different_starting"]:
                #starting_point=list(range(self.params["num_different_starting"]))
                starting_point=random.sample(range(real_length), 
                                      min(self.params["num_different_starting"], real_length)) 
            else:
                starting_point=[0]
        else:
            if self.params["try_different_starting"]:
                starting_point=random.sample(range(real_length), 
                                      min(self.params["num_different_starting"], real_length))
            else:
                starting_point=[random.choice(list(range(real_length)))] # randomly choose one
        # record all molecules from different starting points
        all_mol=[]
        for idx in starting_point: 
            # generate a new molecule
            new_mol, total_log_prob=self.search_and_generate_molecule(idx, np.copy(valences),
                                                sampled_node_symbol, real_length,
                                                random_normal_states, elements, num_vertices)
            # record the molecule with largest number of shapes
            all_mol.append(new_mol)

        # select one out
        #best_mol = select_best(all_mol)
        # nothing generated
        #if best_mol is None:
        #    return
        # visualize it 
        #make_dir('visualization_%s' % dataset)
        #visualize_mol('visualization_%s/%d_%d.png' % (dataset, count, step), best_mol)
        # record the best molecule
        generated_all_similes.extend([Chem.MolToSmiles(mol) for mol in all_mol])
        if len(generated_all_similes) >= self.params['number_of_generation']:
            with open('generated_smiles_%s.txt' % dataset, 'w') as fout:
                for smi in generated_all_similes:
                    fout.write(smi)
                    fout.write("\n")
            print("generation done")
            exit(0)

    def compensate_node_length(self, elements, bucket_size):
        maximum_length=bucket_size+self.params["compensate_num"]
        real_length=get_graph_length([elements['mask']])[0]+self.params["compensate_num"]
        elements['mask']=[1]*real_length + [0]*(maximum_length-real_length)
        elements['init']=np.zeros((maximum_length, self.params["num_symbols"]))
        elements['adj_mat']=np.zeros((self.num_edge_types, maximum_length, maximum_length))
        return maximum_length

    def generate_new_graphs(self, data):
        # bucketed: data organized by bucket
        (bucketed, bucket_sizes, bucket_at_step) = data  
        bucket_counters = defaultdict(int)        
        # all generated similes
        generated_all_similes=[]
        # counter
        count = 0
        # shuffle the lengths
        np.random.shuffle(bucket_at_step)
        for step in range(len(bucket_at_step)):
            bucket = bucket_at_step[step] # bucket number
            # data index
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            # batch data
            elements_batch = bucketed[bucket][start_idx:end_idx]
            for elements in elements_batch:
                # compensate for the length during generation 
                # (this is a result that BFS may not make use of all candidate nodes during generation)
                maximum_length=self.compensate_node_length(elements, bucket_sizes[bucket])
                # initial state
                random_normal_states=generate_std_normal(1, maximum_length,\
                                                         self.params['hidden_size']) # [1, v, h]                
                random_normal_states = self.optimization_over_prior(random_normal_states, 
                                     maximum_length, generated_all_similes,elements, count)
                count+=1
            bucket_counters[bucket] += 1

    def _make_minibatch_iterator_from_file(self, data, is_training: bool):
        bucket_sizes = dataset_info(self.params["dataset"])["bucket_sizes"].copy()
        if is_training:
            np.random.shuffle(bucket_sizes)

        batch_size = self.params["batch_size"]
        for bucket_size in bucket_sizes:
            fname = "{}/data_{}_{:03d}.pkl".format(data, self.params["dataset"], bucket_size)
            print("Loading", fname)
            with open(fname, "rb") as fin:
                bucketed = pickle.load(fin)

            n_samples = len(bucketed)
            indices = np.arange(n_samples)
            if is_training:
                np.random.shuffle(indices)

            if is_training:
                num_batches = n_samples // batch_size
            else:
                num_batches = int(np.ceil(n_samples / batch_size))

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                elements = [bucketed[i] for i in indices[start_idx:end_idx]]
                batch_data = self.make_batch(elements, bucket_size)

                yield self._make_feed_dict(batch_data, len(elements), bucket_size, is_training)

    def _make_minibatch_iterator_from_memory(self, data, is_training: bool):
        (bucketed, bucket_sizes, bucket_at_step) = data
        if is_training:
            np.random.shuffle(bucket_at_step)
            for _, bucketed_data in bucketed.items():
                np.random.shuffle(bucketed_data)

        bucket_counters = defaultdict(int)
        for step in range(len(bucket_at_step)):
            bucket = bucket_at_step[step]
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            elements = bucketed[bucket][start_idx:end_idx]
            bucket_size = bucket_sizes[bucket]
            batch_data = self.make_batch(elements, bucket_size)

            batch_feed_dict = self._make_feed_dict(batch_data, len(elements), bucket_size, is_training)
            bucket_counters[bucket] += 1
            yield batch_feed_dict

    def _make_feed_dict(self, batch_data, batch_size, bucket_size, is_training):
        if is_training:
            dropout_keep_prob = self.params['graph_state_dropout_keep_prob']
            edge_dropout_keep_prob = self.params['edge_weight_dropout_keep_prob']
            out_layer_dropout_keep_prob = self.params['out_layer_dropout_keep_prob']
        else:
            dropout_keep_prob = 1.
            edge_dropout_keep_prob = 1.
            out_layer_dropout_keep_prob = 1.

        initial_representations = batch_data['init']
        initial_representations = self.pad_annotations(initial_representations)
        batch_feed_dict = {
            self.placeholders['initial_node_representation']: initial_representations,
            self.placeholders['node_symbols']: batch_data['init'],
            self.placeholders['latent_node_symbols']: initial_representations,
            self.placeholders['num_graphs']: batch_size,
            self.placeholders['num_vertices']: bucket_size,
            self.placeholders['adjacency_matrix']: batch_data['adj_mat'],
            self.placeholders['node_mask']: batch_data['node_mask'],
            self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
            self.placeholders['edge_weight_dropout_keep_prob']: edge_dropout_keep_prob,
            self.placeholders['iteration_mask']: batch_data['iteration_mask'],
            self.placeholders['incre_adj_mat']: batch_data['incre_adj_mat'],
            self.placeholders['distance_to_others']: batch_data['distance_to_others'],
            self.placeholders['node_sequence']: batch_data['node_sequence'],
            self.placeholders['edge_type_masks']: batch_data['edge_type_masks'],
            self.placeholders['edge_type_labels']: batch_data['edge_type_labels'],
            self.placeholders['edge_masks']: batch_data['edge_masks'],
            self.placeholders['edge_labels']: batch_data['edge_labels'],
            self.placeholders['local_stop']: batch_data['local_stop'],
            self.placeholders['max_iteration_num']: batch_data['max_iteration_num'],
            self.placeholders['kl_trade_off_lambda']: self.params['kl_trade_off_lambda'],
            self.placeholders['out_layer_dropout_keep_prob']: out_layer_dropout_keep_prob,
            self.placeholders['z_prior']: utils.generate_std_normal(
                batch_size, bucket_size, self.params['hidden_size']),
        }
        return batch_feed_dict

if __name__ == "__main__":
    args = docopt(__doc__)
    dataset=args.get('--dataset')
    model = DenseGGNNChemModel(args)
    evaluation = False
    if evaluation:
        model.example_evaluation()
    else:
        model.train()
