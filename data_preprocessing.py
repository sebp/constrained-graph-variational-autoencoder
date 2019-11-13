from os.path import join
import random
import json
import pickle
import numpy as np

from utils import bond_dict, dataset_info, graph_to_adj_mat, to_graph
from data_augmentation import construct_incremental_graph

from joblib import delayed, Parallel


def default_params():
    params = {
        'task_sample_ratios': {},
        'use_edge_bias': True,       # whether use edge bias in gnn

        'clamp_gradient_norm': 1.0,
        'out_layer_dropout_keep_prob': 1.0,

        'tie_fwd_bkwd': True,
        'task_ids': [0],             # id of property prediction

        'random_seed': 0,            # fixed for reproducibility

        'batch_size': 8 if dataset == 'zinc' or dataset == 'cep' else 64,
        "qed_trade_off_lambda": 10,
        'num_epochs': 3 if dataset == 'zinc' or dataset == 'cep' else 10,
        'epoch_to_generate': 9999,  # if dataset=='zinc' or dataset=='cep' else 10,
        'number_of_generation': 50000,
        'maximum_distance': 10,
        # use random sampling or argmax during generation
        "use_argmax_generation": False,
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
        "path_random_order": True,  # False: canonical order, True: random order
        "sample_transition": False,  # whether use transition sampling
        'edge_weight_dropout_keep_prob': 1,
        'check_overlap_edge': False,
        "truncate_distance": 10,
    }

    return params


def graph_to_incremental(graph, starting_idx, chosen_bucket_size, num_edge_types, x_dim, params):
    # Calculate incremental results without master node
    nodes_no_master, edges_no_master = to_graph(
        graph['smiles'], params["dataset"])
    incremental_adj_mat, distance_to_others, node_sequence, edge_type_masks, edge_type_labels, local_stop, edge_masks, edge_labels, overlapped_edge_features =\
        construct_incremental_graph(params["dataset"], edges_no_master, chosen_bucket_size,
                                    len(nodes_no_master), nodes_no_master, params, initial_idx=starting_idx)

    # incremental_result = [incremental_adj_mat, distance_to_others, node_sequence, edge_type_masks,
    #                         edge_type_labels, local_stop, edge_masks, edge_labels, overlapped_edge_features]
    # return incremental_result, graph

    n_active_nodes = len(graph["node_features"])
    bucketed = {
        'adj_mat': graph_to_adj_mat(graph['graph'], chosen_bucket_size, num_edge_types, params['tie_fwd_bkwd']),
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
        'init': graph["node_features"] + [[0 for _ in range(x_dim)] for __ in
                                    range(chosen_bucket_size - n_active_nodes)],
        'labels': [graph["targets"][task_id][0] for task_id in params['task_ids']],
        'mask': [1. for _ in range(n_active_nodes)] + [0. for _ in range(chosen_bucket_size - n_active_nodes)]
    }
    return bucketed



class Preprocessor:

    def __init__(self, params):
        self.params = params

    def _get_starting_indices(self, graph):
        # Use canonical order or random order here. canonical order starts from index 0. random order starts from random nodes
        if not self.params["path_random_order"]:
            # Use several different starting index if using multi BFS path
            if self.params["multi_bfs_path"]:
                list_of_starting_idx = list(
                    range(self.params["bfs_path_count"]))
            else:
                list_of_starting_idx = [0]  # the index 0
        else:
            # get the node length for this molecule
            node_length = len(graph["node_features"])
            if self.params["multi_bfs_path"]:
                list_of_starting_idx = np.random.choice(
                    node_length, self.params["bfs_path_count"], replace=True)  # randomly choose several
            else:
                list_of_starting_idx = [random.choice(
                    list(range(node_length)))]  # randomly choose one
        return list_of_starting_idx

    def calculate_incremental_results(self, raw_data, current_bucket_size, bucket_sizes):
        def filter_by_bucket(graph):
            # choose a bucket
            chosen_bucket_idx = np.argmax(bucket_sizes > max([v for e in graph['graph']
                                                              for v in [e[0], e[2]]]))
            chosen_bucket_size = bucket_sizes[chosen_bucket_idx]
            return chosen_bucket_size == current_bucket_size

        results = Parallel(n_jobs=8, verbose=2)(delayed(graph_to_incremental)(d, starting_idx, current_bucket_size, self.num_edge_types, self.x_dim, self.params)
            for d in filter(filter_by_bucket, raw_data)
            for starting_idx in self._get_starting_indices(d))

        return results

    def run(self, full_path: str, output_dir: str):
        print("Loading data from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)

        # Get some common data out:
        num_fwd_edge_types = len(bond_dict) - 1
        for g in data:
            self.max_num_vertices = max([v for e in g['graph'] for v in [e[0], e[2]]])

        self.num_edge_types = num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2)
        self.annotation_size = len(data[0]["node_features"][0])
        self.x_dim = len(data[0]["node_features"][0])

        bucket_sizes = dataset_info(self.params["dataset"])["bucket_sizes"]
        for bs in bucket_sizes:
            print("Processing graphs in bucket", bs)
            bucketed = self.calculate_incremental_results(
                data, bs, bucket_sizes)

            fname = join(output_dir, "data_{}_{:03d}.pkl".format(self.params["dataset"], bs))
            print("Writing {} graphs to {}".format(len(bucketed), fname))
            with open(fname, "wb") as fin:
                pickle.dump(bucketed, fin, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    dataset = "zinc"
    _params = default_params()
    _params["dataset"] = dataset

    data_files = [
        ("data/molecules_train_zinc.json", "train"),
        ("data/molecules_valid_zinc.json", "valid"),
    ]
    for data_file, output_dir in data_files:
        Preprocessor(_params).run(data_file)
