import networkx as nx
import numpy as np

from abc import ABC, abstractmethod
from typing import FrozenSet, Tuple, Dict, List, Union, Set

from causaldag import DAG
from ..distributions import DistributionBaseClass

class AlgorithmBaseClass(ABC):
    def __init__(self, seed: int) -> None:
        self.random_state = np.random.RandomState(seed)
        self.dag = None
        self.cpdag = None

    def _get_skeleton(self) -> nx.Graph:
        """
        Returns skeleton

        Args:
            None

        Returns:
            G (nx.Graph): Undirected skeleton of DAG
        """
        assert self.dag is not None
        G = nx.Graph()
        G.add_nodes_from(self.dag.nodes)
        G.add_edges_from(self.dag.skeleton)
        return G

    def _get_chain_components(self) -> nx.Graph:                                
        """                                                                     
        Returns chain components in current essential graph                     
                                                                                
        Args:                                                                   
            None                                                                
                                                                                
        Returns:                                                                
            G (nx.Graph): Undirected portions of current essential graph        
        """                                                                     
        assert self.cpdag is not None
        undirected_portions = self.cpdag.copy()                                 
        undirected_portions.remove_all_arcs()                                   
        G = nx.Graph()                                                          
        G.add_nodes_from(undirected_portions.nodes)                             
        G.add_edges_from(undirected_portions.edges)                             
        assert nx.is_chordal(G)
        return G

    @abstractmethod
    def solve(
            self,
            dag: DAG,
            k: int,
            weights: list[float],
            action_distributions: DistributionBaseClass
        ) -> List[FrozenSet[int]]:
        """                                                                     
        Adaptively recover a causally sufficient causal DAG from its essential  
        graph and output the actual interventions used                          
                                                                                
        Note: the search algorithm should only look at the essential graph to   
        decide its next action                                                  
                                                                                
        Args:                                                                   
            dag (DAG): ground truth causal DAG that is causally sufficient      
            k (int): number of actions                                          
            weights (List[int]): weights of each action                         
            action_distributions (DistributionBaseClass):
                Each action corresponds to a DistributionBaseClass object,      
                which we can call .sample() on                                  
                                                                                
        Returns:                                                                
            intervened_nodes (List[FrozenSet[int]]): Collection of              
                interventions that were actually performed                      
        """
        pass

