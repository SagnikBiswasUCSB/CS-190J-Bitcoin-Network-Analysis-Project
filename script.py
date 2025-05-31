import pandas as pd
import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, eye as sparse_eye
from scipy.sparse.linalg import eigs, eigsh
from datetime import datetime
import matplotlib.pyplot as plt
import community as community_louvain # pip install python-louvain
from collections import Counter, defaultdict
import itertools
import os
from sklearn.cluster import KMeans # Added for new algorithm
from sklearn.preprocessing import StandardScaler # Optional: for feature scaling before KMeans

# --- 0. Utility Functions ---
def ensure_dir(directory):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- 1. Plotting Utilities ---
def plot_network_communities(G, partition, title="Network Communities", pos=None, save_path=None, node_size=50, show_labels=False, alpha_nodes=0.8, alpha_edges=0.1, width_edges=0.3, seed=42):
    """Plots the network with nodes colored by community."""
    if G is None or not G.nodes():
        print(f"Plotting Nodes: Graph for '{title}' is empty, cannot plot.")
        return
    if save_path: ensure_dir(os.path.dirname(save_path))
    
    unique_comms = set(partition.values()) if partition and isinstance(partition, dict) else {0}
    num_communities = len(unique_comms)
    
    node_colors = 'skyblue' # Default
    if partition and isinstance(partition, dict) and num_communities > 0:
        cmap = plt.cm.get_cmap('viridis', num_communities)
        node_colors = [cmap(partition.get(node, num_communities)) for node in G.nodes()] # Default for unpartitioned
    elif partition and not isinstance(partition, dict):
            print(f"Plotting Warning: Node partition for '{title}' is not a dictionary. Using default colors.")

    plt.figure(figsize=(15, 15))
    if pos is None:
        if G.number_of_nodes() > 300:
            print(f"Plotting '{title}': {G.number_of_nodes()} nodes. Using spring_layout (seed={seed}, k=0.15, iter=20).")
            pos = nx.spring_layout(G, k=0.15, iterations=20, seed=seed)
        elif G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, seed=seed)
        else: # Empty graph
            plt.title(title + " (Empty Graph)"); plt.axis('off')
            if save_path: plt.savefig(save_path)
            plt.show(); plt.close(); return

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, alpha=alpha_nodes, linewidths=0.2, edgecolors='grey')
    nx.draw_networkx_edges(G, pos, alpha=alpha_edges, width=width_edges, edgelist=list(G.edges())) # Default edge drawing
    
    if show_labels and G.number_of_nodes() < 100:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(title, fontsize=16); plt.axis('off')
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show(); plt.close()

def plot_network_edge_communities(G, edge_to_cluster_map, num_edge_clusters, title="Network Edge Communities", pos=None, save_path=None, node_size=30, node_color='skyblue', alpha_nodes=0.7, width_edges=1.0, seed=42):
    """Plots the network with EDGES colored by community."""
    if G is None or not G.nodes():
        print(f"Plotting Edges: Graph for '{title}' is empty, cannot plot.")
        return
    if not edge_to_cluster_map:
        print(f"Plotting Edges: No edge partition provided for '{title}'. Plotting with default edges.")
        plot_network_communities(G, {}, title=title + " (No Edge Partition)", pos=pos, save_path=save_path, node_size=node_size, show_labels=False, alpha_nodes=alpha_nodes, width_edges=width_edges, seed=seed)
        return

    if save_path: ensure_dir(os.path.dirname(save_path))

    plt.figure(figsize=(17, 17)) # Slightly larger for edge details
    if pos is None:
        if G.number_of_nodes() > 300:
            pos = nx.spring_layout(G, k=0.15, iterations=20, seed=seed)
        elif G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, seed=seed)
        else:
            plt.title(title + " (Empty Graph)"); plt.axis('off');
            if save_path: plt.savefig(save_path)
            plt.show(); plt.close(); return

    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, alpha=alpha_nodes, linewidths=0.1, edgecolors='black')

    if num_edge_clusters > 0 :
        cmap_edges = plt.cm.get_cmap('rainbow', num_edge_clusters) 
        default_edge_color = 'lightgrey'
        edge_colors_list = []
        edgelist_to_draw = []
        
        for u, v, data in G.edges(data=True):
            edge_canonical = tuple(sorted((u,v))) 
            cluster_id = edge_to_cluster_map.get(edge_canonical) 
            if cluster_id is None and G.is_directed(): 
                 cluster_id = edge_to_cluster_map.get( (u,v) )

            if cluster_id is not None:
                edge_colors_list.append(cmap_edges(cluster_id))
            else:
                edge_colors_list.append(default_edge_color) 
            edgelist_to_draw.append((u,v))

        nx.draw_networkx_edges(G, pos, edgelist=edgelist_to_draw, edge_color=edge_colors_list, width=width_edges, alpha=0.6)
    else: 
        nx.draw_networkx_edges(G, pos, width=width_edges, alpha=0.3, edge_color='grey')

    if G.number_of_nodes() < 100: 
        nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(title, fontsize=16); plt.axis('off')
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show(); plt.close()

def plot_degree_distribution(G, title="Degree Distribution", save_path=None, G_is_directed=False):
    """Plots degree (or in/out-degree) distribution. (RETAINED BUT NOT CALLED IN PIPELINE)"""
    if G is None or not G.nodes(): print(f"Degree Plot: Graph for '{title}' is empty."); return
    if save_path: ensure_dir(os.path.dirname(save_path))

    if G_is_directed:
        in_degrees = [G.in_degree(n) for n in G.nodes()]
        out_degrees = [G.out_degree(n) for n in G.nodes()]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(title, fontsize=16)
        if in_degrees:
            axes[0].hist(in_degrees, bins=max(1, min(50, len(set(in_degrees)))), alpha=0.75, color='skyblue', ec='black')
            axes[0].set_title(f"In-Degree Distribution", fontsize=14)
            axes[0].set_xlabel("In-Degree", fontsize=12); axes[0].set_ylabel("Frequency", fontsize=12)
            axes[0].set_yscale('log'); axes[0].set_xscale('log')
        if out_degrees:
            axes[1].hist(out_degrees, bins=max(1, min(50, len(set(out_degrees)))), alpha=0.75, color='salmon', ec='black')
            axes[1].set_title(f"Out-Degree Distribution", fontsize=14)
            axes[1].set_xlabel("Out-Degree", fontsize=12); axes[1].set_ylabel("Frequency", fontsize=12)
            axes[1].set_yscale('log'); axes[1].set_xscale('log')
    else: # Undirected
        degrees = [G.degree(n) for n in G.nodes()]
        if not degrees: print(f"Degree Plot: No degrees to plot for '{title}'."); return
        plt.figure(figsize=(8, 6))
        plt.hist(degrees, bins=max(1, min(50, len(set(degrees)))), alpha=0.75, color='skyblue', ec='black')
        plt.title(title, fontsize=16)
        plt.xlabel("Degree", fontsize=12); plt.ylabel("Frequency", fontsize=12)
        plt.yscale('log'); plt.xscale('log')

    if save_path: plt.savefig(save_path, bbox_inches='tight')
    plt.show(); plt.close()

def plot_eigenvalue_spectrum(eigenvalues, laplacian_name, title_prefix="", save_path=None):
    """Plots the eigenvalue spectrum."""
    if eigenvalues is None or len(eigenvalues) == 0:
        print(f"Spectrum Plot: No eigenvalues to plot for {laplacian_name} of {title_prefix}.")
        return
    if save_path: ensure_dir(os.path.dirname(save_path))
    
    plt.figure(figsize=(8,5))
    plt.plot(np.arange(len(eigenvalues)), eigenvalues.real, 'o-', label=f"Smallest {len(eigenvalues)} eigenvalues")
    plt.title(f"Spectrum of {laplacian_name} for {title_prefix}", fontsize=14)
    plt.xlabel("Eigenvalue Index (Sorted Smallest to Largest)", fontsize=12)
    plt.ylabel("Eigenvalue", fontsize=12)
    plt.legend(); plt.grid(True, linestyle=':', alpha=0.7)
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    plt.show(); plt.close()

def plot_simulated_barcode(simulated_persistence_intervals, dimension_to_plot, filtration_label="Filtration Value (epsilon)", title="Simulated Persistence Barcode", save_path=None):
    """Plots SIMULATED persistence barcodes."""
    if not simulated_persistence_intervals:
        print(f"Simulated Barcode: No intervals for '{title}' Dim {dimension_to_plot}.")
        return
    if save_path: ensure_dir(os.path.dirname(save_path))
    
    plt.figure(figsize=(10, 6))
    bars_this_dim = sorted([item for item in simulated_persistence_intervals if item[1] > item[0] or item[1] == float('inf')], key=lambda x: x[0])
    
    max_finite_death = 0
    has_infinite_bars = False
    processed_bars = []

    for birth, death in bars_this_dim:
        if death == float('inf'):
            processed_bars.append((birth, -1)) 
            has_infinite_bars = True
            max_finite_death = max(max_finite_death, birth)
        elif death > birth: 
            processed_bars.append((birth, death))
            max_finite_death = max(max_finite_death, birth, death)
            
    if not processed_bars:
        print(f"Simulated Barcode: No valid intervals to plot for Dim {dimension_to_plot} in '{title}'.")
        plt.title(title + f" (H{dimension_to_plot} - No Valid Intervals)"); plt.axis('off')
        if save_path: plt.savefig(save_path)
        plt.show(); plt.close(); return

    plot_xlim_upper = max_finite_death * 1.1 if max_finite_death > 0 else 1.0
    if has_infinite_bars and plot_xlim_upper == 0 and processed_bars: 
        plot_xlim_upper = max(b for b,d_val in processed_bars if d_val == -1) * 1.1 if any(d_val == -1 for _,d_val in processed_bars) else 1.0

    for i, (birth, death_val) in enumerate(processed_bars):
        death_to_plot = plot_xlim_upper if death_val == -1 else death_val
        color = 'red' if death_val == -1 else 'blue'
        plt.plot([birth, death_to_plot], [i, i], lw=2, color=color)
    
    plt.yticks([])
    plt.xlabel(filtration_label, fontsize=12)
    min_b_val = min(b for b,d in processed_bars) if processed_bars else 0
    if not processed_bars: 
        min_b_val = 0
        plot_xlim_upper = 1
    elif all(d == -1 for _,d in processed_bars): 
         plot_xlim_upper = max(b for b,d in processed_bars) *1.2 if any(b > 0 for b,d in processed_bars) else 1.0
         if plot_xlim_upper == 0 : plot_xlim_upper = 1.0

    plt.xlim(left=min_b_val, right=plot_xlim_upper if plot_xlim_upper > min_b_val else min_b_val + 1)
    plt.title(title + f" (Simulated H{dimension_to_plot})", fontsize=16)
    if has_infinite_bars or any(item[1] != -1 for item in processed_bars) : 
        legend_elements = []
        if any(item[1] != -1 for item in processed_bars): 
            legend_elements.append(plt.Line2D([0], [0], color='blue', lw=2, label='Finite Interval (Simulated)'))
        if has_infinite_bars:
            legend_elements.append(plt.Line2D([0], [0], color='red', lw=2, label='Infinite Interval (Simulated)'))
        if legend_elements:
             plt.legend(handles=legend_elements, loc='lower right')
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    plt.show(); plt.close()

# --- 2. Data Loading ---
def load_bitcoin_data(csv_path):
    print(f"Loading data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path, header=None, names=['SOURCE', 'TARGET', 'RATING', 'TIME'],
                         compression='gzip' if csv_path.endswith('.gz') else None)
        df['DATETIME'] = pd.to_datetime(df['TIME'], unit='s')
        print(f"  Data loaded: {len(df)} interactions, from {df['DATETIME'].min()} to {df['DATETIME'].max()}.")
        df.dropna(subset=['SOURCE', 'TARGET', 'RATING', 'TIME'], inplace=True)
        df['SOURCE'] = df['SOURCE'].astype(int)
        df['TARGET'] = df['TARGET'].astype(int)
        df['RATING'] = df['RATING'].astype(float)
        df['TIME'] = df['TIME'].astype(int)
        return df
    except FileNotFoundError: 
        print(f"  ERROR: File not found: {csv_path}")
        return None
    except Exception as e: 
        print(f"  ERROR loading data from {csv_path}: {e}")
        return None

# --- 3. Network Construction ---
def build_graph_from_df(df, graph_type=nx.DiGraph, weight_scheme='abs_rating_for_louvain',
                        time_filter_days=None, current_time_dt=None, verbose=True):
    if df is None: return None
    # Corrected line: Use type(graph_type).__name__
    if verbose: print(f"Building graph (type: {type(graph_type).__name__}) with '{weight_scheme}' weights...")
    G = graph_type # graph_type is already an instance e.g. nx.DiGraph()
    if not isinstance(G, (nx.Graph, nx.DiGraph)): # Ensure it's a graph object if passed as a class
        G = graph_type()


    df_to_process = df.copy()
    if time_filter_days is not None:
        if current_time_dt is None: current_time_dt = df_to_process['DATETIME'].max()
        start_time_dt = current_time_dt - pd.Timedelta(days=time_filter_days)
        df_to_process = df_to_process[(df_to_process['DATETIME'] >= start_time_dt) & (df_to_process['DATETIME'] <= current_time_dt)]
        if verbose: print(f"  Filtered data to {len(df_to_process)} interactions from {start_time_dt.date()} to {current_time_dt.date()}.")
    if df_to_process.empty:
        if verbose: print("  No data to build graph after filtering."); return G # Return the empty graph instance
    
    for _, row in df_to_process.iterrows():
        attrs = {'rating': row['RATING'], 'timestamp': row['TIME'], 'datetime': row['DATETIME']}
        final_weight = 0.0
        if weight_scheme == 'abs_rating_for_louvain': final_weight = max(0.00001, abs(row['RATING']))
        elif weight_scheme == 'raw_rating': final_weight = float(row['RATING'])
        elif weight_scheme == 'uniform': final_weight = 1.0
        else: final_weight = max(0.00001, abs(row['RATING']))
        attrs['weight'] = final_weight
        G.add_edge(int(row['SOURCE']), int(row['TARGET']), **attrs)
    
    if verbose: print(f"  Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

# --- 4. Community Detection Algorithms ---
def run_louvain_modularity(G_input, weight_key='weight', plot_title_prefix="", save_plots_dir=None):
    if G_input is None or G_input.number_of_edges() == 0: print(f"Louvain: Graph empty for {plot_title_prefix}."); return {}, 0.0
    
    if G_input.is_directed():
        G_louvain_base = G_input.to_undirected(as_view=False)
    else:
        G_louvain_base = G_input.copy()

    G_louvain = nx.Graph()
    for u, v, data in G_louvain_base.edges(data=True):
        w = data.get(weight_key, 1.0)
        effective_w = abs(w) if w != 0 else 0.00001 
        if G_louvain.has_edge(u,v):
            G_louvain[u][v]['weight'] += effective_w
        else:
            G_louvain.add_edge(u, v, weight=effective_w)
            
    if G_louvain.number_of_edges() == 0: print(f"Louvain: No suitable edges for {plot_title_prefix} after processing."); return {}, 0.0
    
    print(f"Running Louvain for {plot_title_prefix} on {G_louvain.number_of_nodes()} nodes and {G_louvain.number_of_edges()} edges...");
    try:
        partition = community_louvain.best_partition(G_louvain, weight='weight', random_state=42)
        modularity_score = community_louvain.modularity(partition, G_louvain, weight='weight'); 
        num_comms = len(set(partition.values()))
        print(f"  {plot_title_prefix} Louvain: Q={modularity_score:.4f}, Communities={num_comms}")
        if save_plots_dir and G_louvain.number_of_nodes()>0:
            plot_network_communities(G_louvain, partition, title=f"{plot_title_prefix}\nLouvain (Q={modularity_score:.3f})", save_path=f"{save_plots_dir}/{plot_title_prefix}_louvain.png", seed=42)
        return partition, modularity_score
    except Exception as e:
        print(f"  Error running Louvain for {plot_title_prefix}: {e}"); return {}, 0.0

def run_girvan_newman_modularity_tracking(G_input, plot_title_prefix="", save_plots_dir=None, max_nodes_gn=75, num_gn_steps=5):
    if G_input is None or G_input.number_of_edges() == 0: print(f"GN: Graph empty for {plot_title_prefix}."); return {}, 0.0
    
    G_gn_base = nx.Graph(G_input) 
    if G_gn_base.number_of_nodes() == 0: print(f"GN: Graph became empty for {plot_title_prefix}."); return {}, 0.0

    if G_gn_base.number_of_nodes() > max_nodes_gn:
        print(f"GN: Graph for {plot_title_prefix} ({G_gn_base.number_of_nodes()} nodes) too large. Taking largest connected component if smaller or skipping.")
        if not nx.is_connected(G_gn_base):
            largest_cc_nodes = max(nx.connected_components(G_gn_base), key=len)
            G_gn_base = G_gn_base.subgraph(largest_cc_nodes).copy() # Ensure it's a copy
            print(f"  Using largest CC with {G_gn_base.number_of_nodes()} nodes for Girvan-Newman.")
        if G_gn_base.number_of_nodes() > max_nodes_gn:
             print(f"  Largest CC still too large ({G_gn_base.number_of_nodes()} nodes). Skipping GN for {plot_title_prefix}.")
             return {}, 0.0
    
    if G_gn_base.number_of_nodes() == 0 : print(f"GN: Graph empty after CC for {plot_title_prefix}."); return {}, 0.0

    print(f"Running Girvan-Newman for {plot_title_prefix} (up to {num_gn_steps} edge removals)...")
    best_partition_gn = {}
    best_modularity_gn = -1.0 
    
    initial_partition = {node: 0 for node in G_gn_base.nodes()}
    if G_gn_base.number_of_edges() > 0:
        try:
            initial_modularity = community_louvain.modularity(initial_partition, G_gn_base, weight='weight')
        except Exception as e: 
            try:
                initial_modularity = community_louvain.modularity(initial_partition, G_gn_base)
            except Exception as e2:
                print(f" Error calculating initial modularity for GN: {e2}"); initial_modularity = -1
    else: initial_modularity = 0.0
    
    best_modularity_gn = initial_modularity
    best_partition_gn = initial_partition
    
    comp_iter = nx.community.girvan_newman(G_gn_base)
    limited_comp_iter = itertools.islice(comp_iter, num_gn_steps)

    for i, communities_tuple in enumerate(limited_comp_iter):
        partition_map = {}
        for comm_idx, comm_nodes in enumerate(communities_tuple):
            for node in comm_nodes:
                partition_map[node] = comm_idx
        
        current_modularity = -1.0
        if G_gn_base.number_of_edges() > 0: 
            try: 
                current_modularity = community_louvain.modularity(partition_map, G_gn_base, weight='weight')
            except: current_modularity = community_louvain.modularity(partition_map, G_gn_base)

        print(f"  GN Step {i+1}: {len(communities_tuple)} communities, Modularity={current_modularity:.4f}")
        if current_modularity > best_modularity_gn:
            best_modularity_gn = current_modularity
            best_partition_gn = partition_map
            
    print(f"  {plot_title_prefix} Girvan-Newman: Best Q={best_modularity_gn:.4f} with {len(set(best_partition_gn.values()))} communities.")
    if save_plots_dir and best_partition_gn and G_gn_base.number_of_nodes() > 0:
         plot_network_communities(G_gn_base, best_partition_gn, 
                                 title=f"{plot_title_prefix}\nGirvan-Newman (Best Q={best_modularity_gn:.3f})",
                                 save_path=f"{save_plots_dir}/{plot_title_prefix}_girvan_newman.png", seed=42)
    return best_partition_gn, best_modularity_gn

def run_k_clique_communities(G_input, k_clique=3, plot_title_prefix="", save_plots_dir=None):
    if G_input is None or G_input.number_of_nodes() < k_clique: print(f"k-Clique: Graph unsuitable (nodes < k={k_clique}) for {plot_title_prefix}."); return [], {}
    G_kc = nx.Graph(G_input) 
    if G_kc.number_of_nodes() < k_clique: print(f"k-Clique: Graph became unsuitable after making undirected for {plot_title_prefix}."); return [], {}

    print(f"Running {k_clique}-Clique Percolation for {plot_title_prefix}...")
    try:
        comms_list_of_frozensets = list(nx.community.k_clique_communities(G_kc, k_clique))
        comms_list_of_lists = [list(fs) for fs in comms_list_of_frozensets]
        print(f"  {plot_title_prefix} {k_clique}-Clique: Found {len(comms_list_of_lists)} communities.")
        
        node_to_comm_map = {}
        if comms_list_of_lists:
            for comm_idx, nodes_in_comm in enumerate(comms_list_of_lists):
                for node in nodes_in_comm:
                    if node not in node_to_comm_map: 
                        node_to_comm_map[node] = comm_idx
        
        if save_plots_dir and node_to_comm_map and G_kc.number_of_nodes() > 0:
            plot_network_communities(G_kc, node_to_comm_map, 
                                     title=f"{plot_title_prefix}\n{k_clique}-Clique Communities (Colored by First Found)",
                                     save_path=f"{save_plots_dir}/{plot_title_prefix}_kclique_{k_clique}.png", seed=42)
        return comms_list_of_lists, node_to_comm_map
    except Exception as e: print(f"  Error in k-clique for {plot_title_prefix}: {e}"); return [], {}

# --- 5. Hodge Laplacians ---
def get_boundary_operators(G_input, use_absolute_weights_for_structure=False):
    if G_input is None or G_input.number_of_nodes() == 0: return None, None, [], [], [], {}, {}, {}
    
    nodes = sorted(list(G_input.nodes()))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)
    
    structural_edges_tuples = set()
    for u_orig, v_orig in G_input.edges():
        edge = tuple(sorted((u_orig, v_orig), key=lambda n: node_to_idx[n]))
        structural_edges_tuples.add(edge)
    
    oriented_edges_B1 = sorted(list(structural_edges_tuples), key=lambda e: (node_to_idx[e[0]], node_to_idx[e[1]]))
    edge_to_idx_B1 = {edge: i for i, edge in enumerate(oriented_edges_B1)}
    num_edges_B1 = len(oriented_edges_B1)
    
    B1 = None
    if num_nodes > 0 and num_edges_B1 > 0:
        B1 = lil_matrix((num_nodes, num_edges_B1), dtype=np.float32)
        for edge_idx, (u_edge, v_edge) in enumerate(oriented_edges_B1):
            B1[node_to_idx[u_edge], edge_idx] = -1 
            B1[node_to_idx[v_edge], edge_idx] =  1 
        B1 = B1.tocsc(); print(f"  B1 (node-to-edge, structural, canonical orientation) computed. Shape: {B1.shape}")
    else: print("  B1: Not enough nodes or edges for B1 computation.")
    
    temp_G_undirected_for_cliques = nx.Graph()
    temp_G_undirected_for_cliques.add_nodes_from(nodes) 
    temp_G_undirected_for_cliques.add_edges_from(oriented_edges_B1)

    cliques3_node_sets = [frozenset(c) for c in nx.find_cliques(temp_G_undirected_for_cliques) if len(c) == 3]
    
    oriented_triangles = sorted(
        [tuple(sorted(list(t_nodes), key=lambda n: node_to_idx[n])) for t_nodes in set(cliques3_node_sets)],
        key=lambda tpl: (node_to_idx[tpl[0]], node_to_idx[tpl[1]], node_to_idx[tpl[2]])
    )
    triangle_to_idx = {tri: i for i, tri in enumerate(oriented_triangles)}
    num_triangles = len(oriented_triangles)

    B2 = None
    if num_edges_B1 > 0 and num_triangles > 0:
        B2 = lil_matrix((num_edges_B1, num_triangles), dtype=np.float32)
        for t_idx, (n0, n1, n2) in enumerate(oriented_triangles):
            edge_01 = (n0,n1); edge_02 = (n0,n2); edge_12 = (n1,n2)
            
            if edge_12 in edge_to_idx_B1: B2[edge_to_idx_B1[edge_12], t_idx] =  1 
            if edge_02 in edge_to_idx_B1: B2[edge_to_idx_B1[edge_02], t_idx] = -1
            if edge_01 in edge_to_idx_B1: B2[edge_to_idx_B1[edge_01], t_idx] =  1
                
        B2 = B2.tocsc(); print(f"  B2 (edge-to-triangle, structural) computed. Shape: {B2.shape}")
    else: print("  B2: Not enough edges or triangles for B2 computation.")
        
    return B1, B2, nodes, oriented_edges_B1, oriented_triangles, node_to_idx, edge_to_idx_B1, triangle_to_idx

def compute_hodge_laplacians_from_Bs(B1, B2):
    laplacians = {'L0': None, 'L1_down': None, 'L1_up': None, 'L1_full': None, 'L2_down': None}
    if B1 is not None: 
        laplacians['L0'] = B1 @ B1.transpose()
        laplacians['L1_down'] = B1.transpose() @ B1
    
    if B2 is not None:
        laplacians['L1_up'] = B2 @ B2.transpose()
        laplacians['L2_down'] = B2.transpose() @ B2

    if laplacians['L1_down'] is not None and laplacians['L1_up'] is not None:
        if laplacians['L1_down'].shape == laplacians['L1_up'].shape:
            laplacians['L1_full'] = laplacians['L1_down'] + laplacians['L1_up']
        else: 
            print("  Warning: L1_down and L1_up shapes do not match. L1_full will be L1_down.")
            laplacians['L1_full'] = laplacians['L1_down']
    elif laplacians['L1_down'] is not None: 
        laplacians['L1_full'] = laplacians['L1_down']
        print("  L1_up is not available, L1_full is L1_down.")
    elif laplacians['L1_up'] is not None:
        laplacians['L1_full'] = laplacians['L1_up']
        print("  L1_down is not available, L1_full is L1_up.")

    for name,L_matrix in laplacians.items():
        if L_matrix is not None: 
            print(f"  Computed {name}. Shape: {L_matrix.shape}, Non-zero elements: {L_matrix.nnz}")
            if isinstance(L_matrix, lil_matrix): laplacians[name] = L_matrix.tocsc()
    return laplacians

def analyze_laplacian_spectrum(L_matrix, k=6, laplacian_name="L", element_list=None, plot_title_prefix="", save_plots_dir=None):
    if L_matrix is None or L_matrix.shape[0] == 0 : 
        print(f"Spectrum: {laplacian_name} is None or empty for {plot_title_prefix}."); return None, None, None
    if L_matrix.shape[0] < 2 and laplacian_name !="L2_down": 
         print(f"Spectrum: {laplacian_name} too small (shape {L_matrix.shape[0]}) for meaningful eigen-analysis for {plot_title_prefix}."); return None, None, None
    
    print(f"Analyzing spectrum of {laplacian_name} for {plot_title_prefix} (Shape: {L_matrix.shape}, nnz: {L_matrix.nnz})...")
    num_eigen = min(k, L_matrix.shape[0] -1 if L_matrix.shape[0] > 1 else 1) 
    if L_matrix.shape[0] == 1: num_eigen = 1 
    
    if num_eigen <=0: print(f"  Not enough dimensions for eigs on {laplacian_name} (num_eigen={num_eigen})."); return None,None,None
    
    eigenvalues, eigenvectors = None, None
    try:
        if isinstance(L_matrix, lil_matrix): L_csc = L_matrix.tocsc()
        else: L_csc = L_matrix
        
        ncv_val = None # Default for eigsh
        if L_csc.shape[0] > num_eigen + 1: # ncv must be > k and < N for ARPACK. Default is min(N, max(2*k + 1, 20))
             ncv_val = min(L_csc.shape[0]-1, max(2*num_eigen + 1, 20))


        if L_csc.shape[0] == 1: 
            eigenvalues = L_csc.data
            eigenvectors = np.array([[1.0]])
        elif num_eigen < L_csc.shape[0]: 
             eigenvalues, eigenvectors = eigsh(L_csc, k=num_eigen, which='SM', tol=1e-7, ncv=ncv_val) 
        else: # Requesting all eigenvalues for a small matrix (N=k or N=k+1)
             if L_csc.shape[0] > 0 :
                eigenvalues, eigenvectors = np.linalg.eigh(L_csc.toarray()) # dense solver for small N=k case
             else: return None, None, None

        # Sort eigenvalues and corresponding eigenvectors if using np.linalg.eigh
        if num_eigen >= L_csc.shape[0] and L_csc.shape[0] > 0: # if dense solver was used
            sort_indices = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[sort_indices]
            eigenvectors = eigenvectors[:, sort_indices]


        print(f"  Smallest {len(eigenvalues)} eigenvalues of {laplacian_name}: {np.round(eigenvalues.real, 5)}")
        if save_plots_dir: plot_eigenvalue_spectrum(eigenvalues, laplacian_name, title_prefix=plot_title_prefix, save_path=f"{save_plots_dir}/{plot_title_prefix}_{laplacian_name}_spectrum.png")
        
        partition = None
        if laplacian_name == "L0" and len(eigenvalues) >= 2 and eigenvectors is not None and eigenvectors.shape[1] >=2 and element_list is not None and len(element_list) == L_matrix.shape[0]:
            f_vec = eigenvectors[:, 1] 
            partition = {element_list[i]: (1 if f_vec[i] >= 0 else 0) for i in range(len(element_list))}
            lambda2_approx = eigenvalues[1] if len(eigenvalues) > 1 else float('nan')
            print(f"  Fiedler partition from L0 created. lambda_2 approx {lambda2_approx:.5f}")
            
        return eigenvalues, eigenvectors, partition
    except Exception as e: print(f"  Error during {laplacian_name} eigen-decomposition: {e}"); return None, None, None

# --- 5.5 New Algorithm: Hodge L1 Edge Clustering ---
def cluster_elements_from_laplacian_eigenvectors(eigenvectors, element_list, num_clusters=5, laplacian_name="L1_full", n_components_to_use=3, random_state_kmeans=42):
    if eigenvectors is None or len(element_list) != eigenvectors.shape[0]:
        print(f"  Hodge EV Clust: Eigenvectors not suitable or element list mismatch for {laplacian_name}. Skipping."); return {}, None
    if eigenvectors.shape[1] == 0 :
        print(f"  Hodge EV Clust: No eigenvectors available for {laplacian_name}. Skipping."); return {}, None

    k_eig = min(n_components_to_use, eigenvectors.shape[1])
    if k_eig == 0 :
        print(f"  Hodge EV Clust: Not enough eigenvector components (found {eigenvectors.shape[1]}) to use for {laplacian_name}. Skipping."); return {}, None
    
    if laplacian_name == "L0" and eigenvectors.shape[1] > 1 and k_eig > 0 :
        # If using L0 and want to skip the trivial first eigenvector for spectral clustering beyond Fiedler
        # Ensure we don't go out of bounds if only 1 EV component is requested for k_eig after skipping
        if k_eig +1 <= eigenvectors.shape[1] :
             features = eigenvectors[:, 1:k_eig+1]
        elif k_eig == 1 and eigenvectors.shape[1] >=2 : # want 1 component, skip first, take second
             features = eigenvectors[:, 1:2]
        else: # not enough to skip first and take k_eig, so take first k_eig (might include trivial)
             features = eigenvectors[:, :k_eig]
        print(f"  Hodge EV Clust ({laplacian_name}): Using {features.shape[1]} eigenvectors (potentially skipping 1st) for clustering.")


    else: # For L1, L2, etc. or L0 if specified differently
        features = eigenvectors[:, :k_eig]
        print(f"  Hodge EV Clust ({laplacian_name}): Using first {features.shape[1]} eigenvectors for clustering.")

    if features.shape[1] == 0: 
        print(f"  Hodge EV Clust: No features selected after slicing eigenvectors for {laplacian_name}. Skipping."); return {}, None

    print(f"  Running K-Means on {features.real.shape[0]} elements from {laplacian_name} eigenvectors to find {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=min(num_clusters, features.real.shape[0]), # n_clusters cannot be > n_samples
                    random_state=random_state_kmeans, n_init='auto')
    try:
        cluster_labels = kmeans.fit_predict(features.real)
        element_to_cluster_id = {element: label for element, label in zip(element_list, cluster_labels)}
        print(f"  K-Means found {len(set(cluster_labels))} clusters for elements of {laplacian_name}.")
        return element_to_cluster_id, kmeans
    except Exception as e:
        print(f"  Error during K-Means clustering for {laplacian_name}: {e}"); return {}, None

# --- 6. Simulated TDA for Time Filtration (Clique Complex Evolution) ---
def simulate_temporal_clique_complex_ph(G_original_with_timestamps, plot_title_prefix="", save_plots_dir=None, num_time_steps=15, max_simplex_dim_sim=1):
    if G_original_with_timestamps is None or not G_original_with_timestamps.edges(data=True):
        print(f"Simulated Temporal PH: Graph for {plot_title_prefix} is empty or has no timestamped edges."); return {"H0_sim": [], "H1_sim": []}
    
    print(f"Simulating Temporal Clique Complex PH for {plot_title_prefix} over {num_time_steps} snapshots...")
    
    all_edge_timestamps = sorted(list(set(d.get('timestamp') for u,v,d in G_original_with_timestamps.edges(data=True) if 'timestamp' in d)))
    if not all_edge_timestamps:
        print("  Simulated Temporal PH: No timestamps found on edges."); return {"H0_sim": [], "H1_sim": []}

    min_time, max_time = all_edge_timestamps[0], all_edge_timestamps[-1]
    if min_time == max_time: 
        time_points = [min_time]
        num_time_steps = 1
        print(f"  All edges have the same timestamp: {min_time}. Simulating for this single time point.")
    else:
        time_points = np.linspace(min_time, max_time, num_time_steps, dtype=int)

    sim_persistence_h0 = [] 
    sim_persistence_h1 = [] 

    active_h0_features = {} 
    next_h0_feature_id = 0
    node_to_h0_feature_id = {} 

    active_h1_features = [] 
    all_nodes_in_graph = set(G_original_with_timestamps.nodes())
    # node_birth_times = {node: float('inf') for node in all_nodes_in_graph} # Not directly used in this simplified H0
    
    sorted_edges_by_time = sorted(G_original_with_timestamps.edges(data=True), key=lambda x: x[2].get('timestamp', float('inf')))
    
    # G_current_filtration_step = nx.Graph() # Not directly used in this simplified H0
    component_map_at_prev_t = {} 

    for t_idx, current_t_filtration_value in enumerate(time_points):
        G_t = nx.Graph() 
        nodes_at_t = set()
        edges_at_t = []
        for u, v, data in sorted_edges_by_time:
            edge_time = data.get('timestamp')
            if edge_time is not None and edge_time <= current_t_filtration_value:
                nodes_at_t.add(u); nodes_at_t.add(v)
                edges_at_t.append((u,v))
        
        G_t.add_nodes_from(nodes_at_t)
        G_t.add_edges_from(edges_at_t)

        if not G_t.nodes(): continue

        current_components_node_sets = {frozenset(c) for c in nx.connected_components(G_t)}
        
        if t_idx == 0: 
            for comp_nodes in current_components_node_sets:
                component_map_at_prev_t[comp_nodes] = current_t_filtration_value
        else:
            new_component_map_this_t = {}
            # surviving_prev_components = set() # Not strictly needed for this logic
            for prev_comp_nodes, birth_time in component_map_at_prev_t.items():
                merged_into_larger = False
                for curr_comp_nodes in current_components_node_sets:
                    if prev_comp_nodes.issubset(curr_comp_nodes) and prev_comp_nodes != curr_comp_nodes:
                        sim_persistence_h0.append((birth_time, current_t_filtration_value))
                        merged_into_larger = True
                        break
                    elif prev_comp_nodes == curr_comp_nodes: 
                        new_component_map_this_t[curr_comp_nodes] = birth_time
                        # surviving_prev_components.add(prev_comp_nodes)
                        merged_into_larger = True 
                        break
                if not merged_into_larger: 
                     # This means the prev_comp_nodes either vanished (nodes removed - not typical in filtration)
                     # or it's a component that didn't merge and didn't grow but also isn't in current_components_node_sets
                     # (e.g. if graph shrinks, which is not standard filtration).
                     # For a standard filtration, this case (not merged, not survived as is) implies it must have died if its nodes are gone or re-assigned.
                     # Given the snapshot logic, if it's not found, it's considered to have "died" at the previous step's end / this step's start.
                     # However, the current logic only records death upon explicit merge.
                     # To be more robust, if a prev_comp_nodes is not found in new_component_map_this_t after all current_components are processed, it died.
                     # This is implicitly handled as component_map_at_prev_t is overwritten.
                     pass 
            for curr_comp_nodes in current_components_node_sets:
                if curr_comp_nodes in new_component_map_this_t: continue # Already processed as a survivor
                is_continuation_of_one_prev = False
                # Check if this curr_comp_nodes is just a growth of a single previous one that survived
                # (This check is somewhat redundant if the subset logic is comprehensive)
                # The main goal here is to identify genuinely new components.
                # A component is new if it's not a superset of any *single* previous component that would have survived.
                
                # Simpler: if curr_comp_nodes was not formed by merging (handled by prev_comp_nodes.issubset(curr_comp_nodes))
                # and it's not a direct survivor (prev_comp_nodes == curr_comp_nodes), it's "new" or "grown independently".
                # The current logic: if it's not in new_component_map_this_t, it's considered born now.
                new_component_map_this_t[curr_comp_nodes] = current_t_filtration_value
            component_map_at_prev_t = new_component_map_this_t

    for comp_nodes, birth_time in component_map_at_prev_t.items():
        sim_persistence_h0.append((birth_time, float('inf')))
    
    G_cycle_tracker = nx.Graph()
    parent = {node: node for node in all_nodes_in_graph} # Initialize DSU for all potential nodes
    
    # Initialize DSU parent for nodes as they appear if not all are present from start
    # For this simulation, we assume nodes are part of DSU from the beginning if they are in all_nodes_in_graph
    
    def find_set(v_node):
        if v_node not in parent: parent[v_node] = v_node # Initialize if new node encountered
        if v_node == parent[v_node]: return v_node
        parent[v_node] = find_set(parent[v_node])
        return parent[v_node]

    def unite_sets(a_node, b_node):
        if a_node not in parent: parent[a_node] = a_node
        if b_node not in parent: parent[b_node] = b_node
        a_root = find_set(a_node)
        b_root = find_set(b_node)
        if a_root != b_root: parent[b_root] = a_root
    
    num_h1_added = 0
    max_sim_h1 = G_original_with_timestamps.number_of_nodes() // 10 
    
    for u, v, data in sorted_edges_by_time:
        edge_time = data.get('timestamp')
        if edge_time is None: continue

        # Ensure nodes are in DSU before find_set/unite_sets
        if u not in parent: parent[u] = u
        if v not in parent: parent[v] = v
        # G_cycle_tracker.add_node(u); G_cycle_tracker.add_node(v) # Not strictly needed for DSU logic

        if find_set(u) == find_set(v): 
            if num_h1_added < max_sim_h1:
                birth_h1 = edge_time
                death_duration_heuristic = (max_time - birth_h1) * (np.random.uniform(0.1, 0.4)) if max_time > birth_h1 else 0
                death_h1 = birth_h1 + death_duration_heuristic
                death_h1 = min(death_h1, max_time) 
                if death_h1 > birth_h1 : 
                    sim_persistence_h1.append((birth_h1, death_h1))
                    num_h1_added+=1
        else:
            unite_sets(u, v)
        # G_cycle_tracker.add_edge(u,v) # Not strictly needed for DSU based H1 heuristic

    if not sim_persistence_h1 and G_original_with_timestamps.number_of_edges() > G_original_with_timestamps.number_of_nodes() and num_time_steps > 1:
        if min_time < max_time: 
             sim_persistence_h1.append( (min_time + (max_time-min_time)*0.1, max_time - (max_time-min_time)*0.1) )

    print(f"  Simulated Temporal PH: H0 bars: {len(sim_persistence_h0)}, H1 bars (heuristic): {len(sim_persistence_h1)}")
    if save_plots_dir:
        filtration_label_time = f"Time (Timestamp, range {min_time}-{max_time})"
        if sim_persistence_h0:
            plot_simulated_barcode(sim_persistence_h0, 0,
                                   filtration_label=filtration_label_time,
                                   title=f"{plot_title_prefix} Simulated PH (H0, Time Filtration)",
                                   save_path=f"{save_plots_dir}/{plot_title_prefix}_SIM_PH_H0_Time.png")
        else: print(f"No H0 bars to plot for {plot_title_prefix}")
        if sim_persistence_h1:
            plot_simulated_barcode(sim_persistence_h1, 1,
                                   filtration_label=filtration_label_time,
                                   title=f"{plot_title_prefix} Simulated PH (H1, Time Filtration)",
                                   save_path=f"{save_plots_dir}/{plot_title_prefix}_SIM_PH_H1_Time.png")
        else: print(f"No H1 bars to plot for {plot_title_prefix}")
            
    return {"H0_sim": sim_persistence_h0, "H1_sim": sim_persistence_h1}

# --- Main Orchestration Function ---
def main_analysis_pipeline(dataset_file_path, dataset_name, base_save_dir="bitcoin_project_outputs_pipeline"):
    print(f"\n{'='*40}\n RUNNING FULL ANALYSIS PIPELINE FOR: {dataset_name} \n Path: {dataset_file_path}\n{'='*40}")
    ensure_dir(base_save_dir); dataset_results_dir = os.path.join(base_save_dir, dataset_name); ensure_dir(dataset_results_dir)

    df_bitcoin = load_bitcoin_data(dataset_file_path)
    if df_bitcoin is None: print(f"Failed to load data for {dataset_name}. Halting pipeline for this dataset."); return

    # Pass the graph CLASS to build_graph_from_df, not an instance
    G_base_directed_weighted = build_graph_from_df(df_bitcoin, graph_type=nx.DiGraph, weight_scheme='raw_rating', verbose=True)
    
    if G_base_directed_weighted is None or G_base_directed_weighted.number_of_nodes() == 0:
        print(f"Primary graph for {dataset_name} could not be built or is empty. Halting."); return

    G_for_louvain = G_base_directed_weighted 
    G_hodge_input = G_base_directed_weighted 
    G_for_tda_time_filtration = build_graph_from_df(df_bitcoin, graph_type=nx.Graph, weight_scheme='uniform', verbose=False)

    print(f"\n--- Initial Network Visualizations for {dataset_name} ---")
    # plot_degree_distribution(G_base_directed_weighted, title=f"Degree Distribution - {dataset_name}", G_is_directed=True,
    #                          save_path=f"{dataset_results_dir}/{dataset_name}_degree_dist.png") # OMITTED AS REQUESTED
    
    if G_base_directed_weighted.number_of_nodes() < 300:
        plot_network_communities(G_base_directed_weighted, {}, title=f"Network Overview - {dataset_name}", node_size=30,
                                 save_path=f"{dataset_results_dir}/{dataset_name}_network_overview.png", seed=123, show_labels=False)

    print(f"\n--- Community Detection Algorithms for {dataset_name} ---")
    louvain_part, louvain_q = run_louvain_modularity(G_for_louvain, weight_key='weight', 
                                                     plot_title_prefix=dataset_name, save_plots_dir=dataset_results_dir)
    gn_part, gn_q = run_girvan_newman_modularity_tracking(G_for_louvain, plot_title_prefix=dataset_name, 
                                                          save_plots_dir=dataset_results_dir, max_nodes_gn=70, num_gn_steps=5)
    
    G_undirected_for_kclique = nx.Graph(G_base_directed_weighted)
    kclique_comms_list, kclique_node_map = run_k_clique_communities(G_undirected_for_kclique, k_clique=3,
                                                       plot_title_prefix=dataset_name, save_plots_dir=dataset_results_dir)
    if G_undirected_for_kclique.number_of_nodes() > 20 and G_undirected_for_kclique.number_of_edges() > 30 :
         kclique4_comms_list, kclique4_node_map = run_k_clique_communities(G_undirected_for_kclique, k_clique=4,
                                                       plot_title_prefix=dataset_name + "_k4", save_plots_dir=dataset_results_dir)

    print(f"\n--- Hodge Laplacian Analysis for {dataset_name} ---")
    B1, B2, h_nodes, h_edges, h_triangles, node_to_idx, edge_to_idx, tri_to_idx = get_boundary_operators(G_hodge_input)
    all_laplacians = compute_hodge_laplacians_from_Bs(B1, B2)
    
    L0_spec, L0_vecs, fiedler_part_L0 = None, None, None
    if all_laplacians.get('L0') is not None:
        L0_spec, L0_vecs, fiedler_part_L0 = analyze_laplacian_spectrum(all_laplacians['L0'], k=6, laplacian_name="L0",
                                                                  element_list=h_nodes, plot_title_prefix=dataset_name, 
                                                                  save_plots_dir=dataset_results_dir)
        if fiedler_part_L0 and G_base_directed_weighted.number_of_nodes() > 0 :
            G_plot_fiedler = nx.Graph(G_hodge_input) 
            lambda2_val = L0_spec[1] if L0_spec is not None and len(L0_spec) > 1 else 'N/A'
            title_fiedler = f"{dataset_name}\nFiedler L0 Partition ($\\lambda_2 \\approx {lambda2_val:.3f}$)"
            plot_network_communities(G_plot_fiedler, fiedler_part_L0,
                                     title=title_fiedler,
                                     save_path=f"{dataset_results_dir}/{dataset_name}_Fiedler_L0_Partition.png", seed=123)
    
    L1f_spec, L1f_vecs, _ = None, None, None
    if all_laplacians.get('L1_full') is not None:
        L1f_spec, L1f_vecs, _ = analyze_laplacian_spectrum(all_laplacians['L1_full'], k=max(6, n_ev_components_for_L1_clust if 'n_ev_components_for_L1_clust' in locals() else 6), # Ensure enough EVs for clustering
                                                      laplacian_name="L1_full",
                                                      element_list=h_edges, plot_title_prefix=dataset_name, save_plots_dir=dataset_results_dir)
    
    L2d_spec, L2d_vecs, _ = None, None, None
    if all_laplacians.get('L2_down') is not None: 
         L2d_spec, L2d_vecs, _ = analyze_laplacian_spectrum(all_laplacians['L2_down'], k=min(6, len(h_triangles) if h_triangles else 1), 
                                                      laplacian_name="L2_down",
                                                      element_list=h_triangles, plot_title_prefix=dataset_name, save_plots_dir=dataset_results_dir)

    print(f"\n--- Hodge L1 Edge Community Detection for {dataset_name} ---")
    num_edge_clusters = 5 
    n_ev_components_for_L1_clust = 3 
    if L1f_vecs is not None and h_edges:
        edge_cluster_partition, kmeans_model_L1 = cluster_elements_from_laplacian_eigenvectors(
            L1f_vecs, h_edges, 
            num_clusters=num_edge_clusters, 
            laplacian_name="L1_full_edges",
            n_components_to_use=n_ev_components_for_L1_clust
        )
        
        if edge_cluster_partition:
            print(f"  Found {len(set(edge_cluster_partition.values()))} edge communities using L1 eigenvectors.")
            plot_G_for_edge_comms = nx.Graph(G_hodge_input)
            
            plot_network_edge_communities(plot_G_for_edge_comms, edge_cluster_partition, len(set(edge_cluster_partition.values())),
                                          title=f"{dataset_name}\nEdge Communities from $L_1$ Eigenvectors ({num_edge_clusters} clusters)",
                                          save_path=f"{dataset_results_dir}/{dataset_name}_L1_Edge_Communities.png",
                                          seed=123)
        else:
            print(f"  Edge clustering from L1 eigenvectors failed or produced no partition for {dataset_name}.")
    else:
        print(f"  Skipping L1 Edge Clustering for {dataset_name} due to missing L1 eigenvectors or edges.")

    print(f"\n--- Simulated Topological Data Analysis (Time Filtration) for {dataset_name} ---")
    sim_ph_results_time = simulate_temporal_clique_complex_ph(G_for_tda_time_filtration, 
                                                        plot_title_prefix=dataset_name,
                                                        save_plots_dir=dataset_results_dir,
                                                        num_time_steps=20, 
                                                        max_simplex_dim_sim=1)
    print(f"  Simulated PH (Time Filtration) for {dataset_name}: H0 found {len(sim_ph_results_time.get('H0_sim',[]))} bars, H1 found {len(sim_ph_results_time.get('H1_sim',[]))} bars.")
    
    print(f"\n{'='*40}\n FINISHED ANALYSIS PIPELINE FOR: {dataset_name} \n Outputs in: {dataset_results_dir} \n{'='*40}")

if __name__ == "__main__":
    MAIN_OUTPUT_DIR = "Bitcoin_Project_Analysis_Outputs_Pipeline_Run" 
    ensure_dir(MAIN_OUTPUT_DIR)

    BITCOIN_OTC_CSV_GZ = "soc-sign-bitcoinotc.csv.gz"
    BITCOIN_ALPHA_CSV_GZ = "soc-sign-bitcoinalpha.csv.gz"

    datasets_to_run = {}
    if os.path.exists(BITCOIN_OTC_CSV_GZ):
        datasets_to_run["BitcoinOTC"] = BITCOIN_OTC_CSV_GZ
    else:
        print(f"ERROR: Data file not found: '{BITCOIN_OTC_CSV_GZ}'. Please download it from SNAP (https://snap.stanford.edu/data/soc-sign-bitcoinotc.html) and place it in the script directory or provide the full path.")

    if os.path.exists(BITCOIN_ALPHA_CSV_GZ):
        datasets_to_run["BitcoinAlpha"] = BITCOIN_ALPHA_CSV_GZ
    else:
         print(f"ERROR: Data file not found: '{BITCOIN_ALPHA_CSV_GZ}'. Please download it from SNAP (https://snap.stanford.edu/data/soc-sign-bitcoinalpha.html) and place it in the script directory or provide the full path.")

    if not datasets_to_run:
        print("No datasets found. Exiting.")
    else:
        for name, path in datasets_to_run.items():
            main_analysis_pipeline(path, name, base_save_dir=MAIN_OUTPUT_DIR)
    
    print(f"\n\nAll analyses complete. Outputs are in '{MAIN_OUTPUT_DIR}'.")
    print("Reminder on SIMULATED TDA: Barcodes from TDA are illustrative simulations of the process and output type based on graph component evolution and cycle heuristics. Rigorous PH requires specialized libraries.")
    print("Reminder on HODGE LAPLACIANS: B1 and B2 construction is structural (unweighted incidence). Weighted Hodge Laplacians or those on directed complexes are more advanced variations.")
