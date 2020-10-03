import torch
import utils as u
import os


class AMLDataset():

    def __init__(self, args):
        assert args.task in ['link_pred',
                             'edge_cls'], 'bitcoin only implements link_pred or edge_cls'
        self.ecols = u.Namespace({
            'orig_acct': 0,
            'bene_acct': 1,
            'base_amt': 2,
            'tran_timestamp': 3,
            'is_sar': 4
        })
        args.amlsim_args = u.Namespace(args.amlsim_args)

        #build edge data structure
        edges = self.load_edges(args.amlsim_args)

        edges = self.make_contigous_node_ids(edges)
        num_nodes = edges[:, [self.ecols.orig_acct, self.ecols.bene_acct]].unique().size(0)

        tran_timestamps = u.aggregate_by_time(edges[:, self.ecols.tran_timestamp],
                                              args.amlsim_args.aggr_time)
        self.max_time = tran_timestamps.max()
        self.min_time = tran_timestamps.min()
        edges[:, self.ecols.tran_timestamp] = tran_timestamps

        #add the reversed link to make the graph undirected
        edges = torch.cat([
            edges, edges[:, [
                self.ecols.bene_acct, self.ecols.orig_acct, self.ecols.base_amt, self.ecols.
                tran_timestamp, self.ecols.is_sar
            ]]
        ])

        #separate classes
        sp_indices = edges[:, [
            self.ecols.orig_acct, self.ecols.bene_acct, self.ecols.tran_timestamp, self.ecols.
            base_amt
        ]].t()
        sp_values = edges[:, self.ecols.is_sar]

        neg_mask = sp_values == 0

        neg_sp_indices = sp_indices[:, neg_mask]
        neg_sp_values = sp_values[neg_mask]
        neg_sp_edges = torch.sparse.LongTensor(
            neg_sp_indices, neg_sp_values,
            torch.Size([num_nodes, num_nodes, num_nodes, self.max_time + 1])).coalesce()

        pos_mask = sp_values == 1

        pos_sp_indices = sp_indices[:, pos_mask]
        pos_sp_values = sp_values[pos_mask]

        pos_sp_edges = torch.sparse.LongTensor(
            pos_sp_indices, pos_sp_values, torch.Size([num_nodes, num_nodes,
                                                       self.max_time + 1])).coalesce()

        #scale positive class to separate after adding
        pos_sp_edges *= 1000

        #we substract the neg_sp_edges to make the values positive
        sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()

        #separating negs and positive edges per edge/timestamp
        vals = sp_edges._values()
        neg_vals = vals % 1000
        pos_vals = vals // 1000
        #We add the negative and positive scores and do majority voting
        vals = pos_vals - neg_vals
        #creating is_sars new_vals -> the is_sar of the edges
        new_vals = torch.zeros(vals.size(0), dtype=torch.long)
        new_vals[vals > 0] = 1
        new_vals[vals <= 0] = 0
        indices_is_sars = torch.cat([sp_edges._indices().t(), new_vals.view(-1, 1)], dim=1)

        #the base_amt of the edges (vals), is simply the number of edges between two entities at each time_step
        vals = pos_vals + neg_vals

        self.edges = {'idx': indices_is_sars, 'vals': vals}
        self.num_nodes = num_nodes
        self.num_classes = 2

    def prepare_node_feats(self, node_feats):
        node_feats = node_feats[0]
        return node_feats

    def edges_to_sp_dict(self, edges):
        idx = edges[:, [self.ecols.orig_acct, self.ecols.bene_acct, self.ecols.tran_timestamp]]

        vals = edges[:, self.ecols.base_amt]
        return {'idx': idx, 'vals': vals}

    def get_num_nodes(self, edges):
        all_ids = edges[:, [self.ecols.orig_acct, self.ecols.bene_acct]]
        num_nodes = all_ids.max() + 1
        return num_nodes

    def load_edges(self, amlsim_args):
        file = os.path.join(amlsim_args.folder, amlsim_args.edges_file)
        with open(file) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines]
        edges = torch.tensor(edges, dtype=torch.long)
        return edges

    def make_contigous_node_ids(self, edges):
        new_edges = edges[:, [self.ecols.orig_acct, self.ecols.bene_acct]]
        _, new_edges = new_edges.unique(return_inverse=True)
        edges[:, [self.ecols.orig_acct, self.ecols.bene_acct]] = new_edges
        return edges
