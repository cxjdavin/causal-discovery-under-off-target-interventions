import bnlearn as bn
import networkx as nx
import os
import sys

def bif_to_nx(bif_dirname, nx_dirname):
    for bif_fname in os.listdir(bif_dirname):
        bif_model = bn.import_DAG("{0}/{1}".format(bif_dirname, bif_fname))
        nx_dag = nx.from_pandas_adjacency(bif_model['adjmat'],
                                          create_using=nx.DiGraph)
        nx_dag = nx.convert_node_labels_to_integers(nx_dag)
        fname = bif_fname.split('.')[0]
        nx.write_adjlist(nx_dag, "{0}/{1}.nx".format(nx_dirname, fname))

if __name__ == "__main__":
    bif_dirname = sys.argv[1]
    nx_dirname = sys.argv[2]
    assert os.path.exists(bif_dirname)
    os.makedirs(nx_dirname, exist_ok=True)
    bif_to_nx(bif_dirname, nx_dirname)

