import networkx as nx
import numpy as np

def LFR_benchmark_graph():
    n = 1000  #节点数
    for mu in np.arange(0,1,0.05):
        G = nx.LFR_benchmark_graph(n, tau1=2.5, tau2=2.0, mu=mu, 
                                   average_degree=None, min_degree=None, max_degree=None, 
                                   min_community=None, max__community=None, max_community=None, 
                                   tol=1e-07, max_iters=500, seed=None)


    
    pass



if __name__=='__main__':
    G = LFR_benchmark_graph()