import networkx as nx
import community as community_louvain
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
import numpy as np
from itertools import combinations


def LFR_benchmark_graph():
    N = 1000  #节点数		

    for mu in np.arange(0.1,0.6,0.1):
        #LFR benchmark生成网络
        G = nx.LFR_benchmark_graph(n=N, tau1=3.0, tau2=1.5, mu=mu, 
                                   average_degree=5, min_degree=None, max_degree=None, 
                                   min_community=20, max_community=30, 
                                   tol=1e-07, max_iters=500, seed=42)
        communities = {node: data["community"] for node, data in G.nodes(data=True)}
        true_partitions = np.zeros(N)
        for node, comms in communities.items():
            true_partitions[node] = list(comms)[0]
        #CNM算法
        cnm_communities = list(greedy_modularity_communities(G))
        cnm_partitions = {node: i for i, com in enumerate(cnm_communities) for node in com}
        #Louvain算法
        louvain_partitions = community_louvain.best_partition(G)
        louvain_communities = {}
        for node, community_id in louvain_partitions.items():
            louvain_communities.setdefault(community_id, []).append(node)
        louvain_communities = list(louvain_communities.values())
        #评价CNM算法性能
        cnm_results = evaluate_performance(G, cnm_communities, cnm_partitions)
        print("CNM Evaluation: Modularity: {}, Coverage: {}, Performance: {}, Rand Index: {}, NMI: {}".format(*cnm_results))
        #评价Louvain算法性能
        louvain_results = evaluate_performance(G, louvain_communities, louvain_partitions)
        print("Louvain Evaluation: Modularity: {}, Coverage: {}, Performance: {}, Rand Index: {}, NMI: {}".format(*louvain_results))
        #保存结果
        save_result(mu, cnm_results, louvain_results)
        save_graph(G, mu)

def save_graph(G, mu):
    #保存生成网络的.net文件
    nx.write_pajek(G, f"lfr_network_mu_{mu:.2f}.net")

def save_result(mu, cnm_results, louvain_results):
    with open(f'./results_mu_{mu:.2f}.txt', 'w') as f:
        f.write(f"mu: {mu}\n")
        f.write("CNM Evaluation:\n")
        f.write(f"Modularity: {cnm_results[0]}, Coverage: {cnm_results[1]}, Performance: {cnm_results[2]}, Rand Index: {cnm_results[3]}, NMI: {cnm_results[4]}\n")
        f.write("Louvain Evaluation:\n")
        f.write(f"Modularity: {louvain_results[0]}, Coverage: {louvain_results[1]}, Performance: {louvain_results[2]}, Rand Index: {louvain_results[3]}, NMI: {louvain_results[4]}\n")

def evaluate_performance(G, true_partitions, detected_partitions):
    #模块度
    modularity = community_louvain.modularity(detected_partitions, G)
    #Coverage and Performance
    coverage, performance = nx.algorithms.community.quality.partition_quality(G, true_partitions)
    #Rand index
    node_to_community = {node: idx for idx, community in enumerate(true_partitions) for node in community}
    true_labels = [node_to_community[node] for node in G.nodes()]
    predicted_labels = [G.nodes[node]['community'] for node in G.nodes()]
    rand_index = adjusted_rand_score(true_labels, predicted_labels)
    #NMI
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    #返回评价指标 
    return modularity, coverage, performance, rand_index, nmi



if __name__=='__main__':
    G = LFR_benchmark_graph()