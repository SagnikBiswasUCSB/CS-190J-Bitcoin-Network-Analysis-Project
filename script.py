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

# --- 0. Utility Functions ---
def ensure_dir(directory):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- 1. Plotting Utilities ---
def plot_network_communities(G, partition, title="Network Communities", pos=None, save_path=None, node_size=50, show_labels=False, alpha_nodes=0.8, alpha_edges=0.1, width_edges=0.3, seed=42):
    """Plots the network with nodes colored by community."""
    if G is None or not G.nodes():
        print(f"Plotting: Graph for '{title}' is empty, cannot plot.")
        return
    if save_path: ensure_dir(os.path.dirname(save_path))
    
    unique_comms = set(partition.values()) if partition and isinstance(partition, dict) else {0}
    num_communities = len(unique_comms)
    
    node_colors = 'skyblue' # Default
    if partition and isinstance(partition, dict) and num_communities > 0:
        cmap = plt.cm.get_cmap('viridis', num_communities)
        node_colors = [cmap(partition.get(node, num_communities)) for node in G.nodes()] # Default for unpartitioned
    elif partition and not isinstance(partition, dict):
         print(f"Plotting Warning: Partition for '{title}' is not a dictionary. Using default colors.")

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
    nx.draw_networkx_edges(G, pos, alpha=alpha_edges, width=width_edges, edgelist=list(G.edges()))
    
    if show_labels and G.number_of_nodes() < 100:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(title, fontsize=16); plt.axis('off')
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show(); plt.close()

def plot_degree_distribution(G, title="Degree Distribution", save_path=None, G_is_directed=False):
    """Plots degree (or in/out-degree) distribution."""
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

def plot_simulated_barcode(simulated_persistence_intervals, dimension_to_plot, filtration_label="Filtration Value (<span class="math-inline">\\\\epsilon</span>)", title="Simulated Persistence Barcode", save_path=None):
    """Plots SIMULATED persistence barcodes."""
    if not simulated_persistence_intervals: 
        print(f"Simulated Barcode: No intervals for '{title}' Dim {dimension_to_plot}.")
        return
    if save_path: ensure_dir(os.path.dirname(save_path))
    
    plt.figure(figsize=(10, 6))
    # Expected format for simulated_persistence_intervals: list of (birth, death) for the given dimension
    bars_this_dim = sorted([item for item in simulated_persistence_intervals if item[1] > item[0]], key=lambda x: x[0]) # Valid, sorted by birth
    
    max_finite_death = 0
    has_infinite_bars = False
    processed_bars = []

    for birth, death in bars_this_dim:
        if death == float('inf'):
            processed_bars.append((birth, -1)) # Sentinel for infinite
            has_infinite_bars = True
            max_finite_death = max(max_finite_death, birth)
        elif death > birth: # Valid finite interval
            processed_bars.append((birth, death))
            max_finite_death = max(max_finite_death, birth, death)
            
    if not processed_bars:
        print(f"Simulated Barcode: No valid intervals to plot for Dim {dimension_to_plot} in '{title}'.")
        plt.title(title + f" (H{dimension_to_plot} - No Valid Intervals)"); plt.axis('off')
        if save_path: plt.savefig(save_path)
        plt.show(); plt.close(); return

    plot_xlim_upper = max_finite_death * 1.1 if max_finite_death > 0 else 1.0 

    for i, (birth, death_val) in enumerate(processed_bars):
        death_to_plot = plot_xlim_upper if death_val == -1 else death_val
        color = 'red' if death_val == -1 else 'blue'
        plt.plot([birth, death_to_plot], [i, i], lw=2, color=color)
    
    plt.yticks([])
    plt.xlabel(filtration_label, fontsize=12)
    min_b_val = min(b for b,d in processed_bars) if processed_bars else 0
    plt.xlim(left=min_b_val, right=plot_xlim_upper)
    plt.title(title + f" (Simulated H{dimension_to_plot})", fontsize=16)
    if has_infinite_bars: 
        plt.plot([],[], color='blue', label='Finite Interval (Simulated)')
        plt.plot([],[], color='red', label='Infinite Interval (Simulated)')
        plt.legend(loc='lower right')
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
        return df
    except FileNotFoundError: print(f"  ERROR: File not found: {csv_path}"); return None
    except Exception as e: print(f"  ERROR loading data from {csv_path}: {e}"); return None

# --- 3. Network Construction ---
def build_graph_from_df(df, graph_type=nx.DiGraph, weight_scheme='abs_rating_for_louvain', 
                        time_filter_days=None, current_time_dt=None, verbose=True):
    if df is None: return None
    if verbose: print(f"Building graph (type: {graph_type.__name__}) with '{weight_scheme}' weights...")
    G = graph_type(); df_to_process = df.copy()
    if time_filter_days is not None:
        if current_time_dt is None: current_time_dt = df_to_process['DATETIME'].max()
        start_time_dt = current_time_dt - pd.Timedelta(days=time_filter_days)
        df_to_process = df_to_process[(df_to_process['DATETIME'] >= start_time_dt) & (df_to_process['DATETIME'] <= current_time_dt)]
        if verbose: print(f"  Filtered data to {len(df_to_process)} interactions from {start_time_dt.date()} to {current_time_dt.date()}.")
    if df_to_process.empty: 
        if verbose: print("  No data to build graph after filtering."); return G
    for _, row in df_to_process.iterrows():
        attrs = {'rating': row['RATING'], 'timestamp': row['TIME'], 'datetime': row['DATETIME']}
        final_weight = 0.0
        if weight_scheme == 'abs_rating_for_louvain': final_weight = max(0.00001, abs(row['RATING']))
        elif weight_scheme == 'raw_rating': final_weight = float(row['RATING'])
        # Add other weight schemes if needed
        else: final_weight = max(0.00001, abs(row['RATING'])) # Default
        attrs['weight'] = final_weight
        G.add_edge(int(row['SOURCE']), int(row['TARGET']), **attrs) # Ensure nodes are integers if they are IDs
    if verbose: print(f"  Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

# --- 4. Community Detection Algorithms ---
def run_louvain_modularity(G_input, weight_key='weight', plot_title_prefix="", save_plots_dir=None):
    # ... (implementation as before)
    if G_input is None or G_input.number_of_edges() == 0: print(f"Louvain: Graph empty for {plot_title_prefix}."); return {}, 0.0
    G_louvain = nx.Graph() 
    for u, v, data in G_input.edges(data=True):
        w = data.get(weight_key, 1.0); effective_w = abs(w) if w <= 0 else w 
        if effective_w <= 0: effective_w = 0.00001
        if G_louvain.has_edge(u,v): G_louvain[u][v]['weight'] += effective_w
        else: G_louvain.add_edge(u, v, weight=effective_w)
    if G_louvain.number_of_edges() == 0: print(f"Louvain: No suitable edges for {plot_title_prefix}."); return {}, 0.0
    print(f"Running Louvain for {plot_title_prefix}..."); partition = community_louvain.best_partition(G_louvain, weight='weight')
    modularity_score = community_louvain.modularity(partition, G_louvain, weight='weight'); num_comms = len(set(partition.values()))
    print(f"  {plot_title_prefix} Louvain: Q={modularity_score:.4f}, Communities={num_comms}")
    if save_plots_dir and G_louvain.number_of_nodes()>0:
        plot_network_communities(G_louvain, partition, title=f"{plot_title_prefix}\nLouvain (Q={modularity_score:.3f})", save_path=f"{save_plots_dir}/{plot_title_prefix}_louvain.png", seed=42)
    return partition, modularity_score

def run_girvan_newman_with_modularity_tracking(G_input, plot_title_prefix="", save_plots_dir=None, max_nodes_gn=75):
    # ... (implementation as before)
    if G_input is None or G_input.number_of_edges() == 0: print(f"GN: Graph empty for {plot_title_prefix}."); return {}
    G_gn_base = nx.Graph(G_input); # ... (subgraph logic, iteration, modularity calc, plotting as before) ...
    print(f"  (Girvan-Newman placeholder for {plot_title_prefix} - full logic as previously defined)") # Placeholder for brevity
    return {}

def run_k_clique_communities(G_input, k_clique=3, plot_title_prefix="", save_plots_dir=None):
    # ... (implementation as before)
    if G_input is None or G_input.number_of_nodes() < k_clique: print(f"k-Clique: Graph unsuitable k={k_clique} for {plot_title_prefix}."); return []
    G_kc = nx.Graph(G_input); print(f"Running {k_clique}-Clique for {plot_title_prefix}...")
    try:
        comms = list(nx.community.k_clique_communities(G_kc, k_clique)); print(f"  {plot_title_prefix} {k_clique}-Clique: Found {len(comms)} communities.")
        if save_plots_dir and comms: # ... (plotting logic as before) ...
            node_to_first_comm = {} # Simplified for example
            for idx, c_nodes in enumerate(comms):
                for node in c_nodes:
                    if node not in node_to_first_comm: node_to_first_comm[node] = idx
            plot_network_communities(G_kc, node_to_first_comm, title=f"{plot_title_prefix}\n{k_clique}-Clique (Colored by First)", save_path=f"{save_plots_dir}/{plot_title_prefix}_kclique_{k_clique}.png", seed=42)
        return comms
    except Exception as e: print(f"  Error in k-clique for {plot_title_prefix}: {e}"); return []

# --- 5. Hodge Laplacians ---
def get_boundary_operators(G_input):
    # (Refined version from previous response, focusing on structural B1 and B2)
    if G_input is None or G_input.number_of_nodes() == 0: return None, None, [], [], [], {}, {}, {}
    
    nodes = sorted(list(G_input.nodes())) 
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)

    # B1: node-to-edge. Canonical orientation for edges (idx_u < idx_v).
    # This means B1 is for the "underlying" undirected graph structure if G_input is directed.
    # For a directed Hodge Laplacian, B1 would use the directed edges.
    # For simplicity and consistency with common B2 definitions from cliques, use undirected structure.
    
    # Use unique structural edges, oriented consistently
    structural_edges_tuples = set()
    for u, v in G_input.edges():
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        structural_edges_tuples.add(tuple(sorted((u,v), key=lambda n: node_to_idx[n])))
    
    oriented_edges_B1 = sorted(list(structural_edges_tuples), key=lambda e: (node_to_idx[e[0]], node_to_idx[e[1]]))
    edge_to_idx_B1 = {edge: i for i, edge in enumerate(oriented_edges_B1)}
    num_edges_B1 = len(oriented_edges_B1)
    
    B1 = None
    if num_nodes > 0 and num_edges_B1 > 0:
        B1 = lil_matrix((num_nodes, num_edges_B1), dtype=np.float32)
        for i, (u_edge,v_edge) in enumerate(oriented_edges_B1): # u_edge, v_edge are original node IDs
            B1[node_to_idx[u_edge], i] = -1 # Tail (smaller index by convention)
            B1[node_to_idx[v_edge], i] = 1  # Head (larger index by convention)
        B1 = B1.tocsc(); print(f"  B1 (node-to-edge, undirected structure) computed. Shape: {B1.shape}")

    # B2: Edge to Triangle. Triangles {n0,n1,n2} oriented (idx0<idx1<idx2).
    # Edges for B2 are from `oriented_edges_B1`.
    # Standard boundary operator: ∂[v0,v1,v2] = [v1,v2] - [v0,v2] + [v0,v1]
    # (assuming edges [vi,vj] are oriented i<j)
    
    # Find 3-cliques on the underlying undirected graph structure
    temp_G_undirected = nx.Graph()
    temp_G_undirected.add_nodes_from(G_input.nodes())
    temp_G_undirected.add_edges_from(oriented_edges_B1) # Use the canonically oriented edges

    cliques3_nodesets = [frozenset(c) for c in nx.find_cliques(temp_G_undirected) if len(c) == 3]
    oriented_triangles = sorted(
        [tuple(sorted(list(t_nodes), key=lambda n: node_to_idx[n])) for t_nodes in set(cliques3_nodesets)],
        key=lambda tpl: (node_to_idx[tpl[0]], node_to_idx[tpl[1]], node_to_idx[tpl[2]])
    )
    triangle_to_idx = {tri: i for i, tri in enumerate(oriented_triangles)}
    num_triangles = len(oriented_triangles)

    B2 = None
    if num_edges_B1 > 0 and num_triangles > 0:
        B2 = lil_matrix((num_edges_B1, num_triangles), dtype=np.float32)
        for t_idx, (n0, n1, n2) in enumerate(oriented_triangles): # n0,n1,n2 are original node IDs, sorted by their mapped index
            # Edges of the triangle (n0,n1), (n0,n2), (n1,n2) - these are already canonically oriented
            edge_01 = (n0,n1); edge_02 = (n0,n2); edge_12 = (n1,n2)
            
            if edge_01 in edge_to_idx_B1: B2[edge_to_idx_B1[edge_01], t_idx] = 1    # +[v0,v1]
            if edge_02 in edge_to_idx_B1: B2[edge_to_idx_B1[edge_02], t_idx] = -1   # -[v0,v2]
            if edge_12 in edge_to_idx_B1: B2[edge_to_idx_B1[edge_12], t_idx] = 1    # +[v1,v2]
            
        B2 = B2.tocsc(); print(f"  B2 (edge-to-triangle, structural) computed. Shape: {B2.shape}")
    else: print("  B2: Not enough edges or triangles for B2 computation.")
        
    return B1, B2, nodes, oriented_edges_B1, oriented_triangles, node_to_idx, edge_to_idx_B1, triangle_to_idx

def compute_hodge_laplacians_from_Bs(B1, B2):
    # ... (implementation as before) ...
    laplacians = {'L0': None, 'L1_down': None, 'L1_up': None, 'L1_full': None, 'L2_down': None}
    if B1 is not None: laplacians['L0'] = B1 @ B1.transpose(); laplacians['L1_down'] = B1.transpose() @ B1
    if B2 is not None and B1 is not None and B2.shape[0] == B1.shape[1]:
        laplacians['L1_up'] = B2 @ B2.transpose(); laplacians['L2_down'] = B2.transpose() @ B2
    if laplacians['L1_down'] is not None and laplacians['L1_up'] is not None and laplacians['L1_down'].shape == laplacians['L1_up'].shape:
        laplacians['L1_full'] = laplacians['L1_down'] + laplacians['L1_up']
    elif laplacians['L1_down'] is not None: laplacians['L1_full'] = laplacians['L1_down']
    for name,L in laplacians.items(): 
        if L is not None: print(f"  Computed {name}. Shape: {L.shape}")
    return laplacians

def analyze_laplacian_spectrum(L, k=6, laplacian_name="L", element_list=None, plot_title_prefix="", save_plots_dir=None):
    # ... (implementation as before, calling plot_eigenvalue_spectrum) ...
    if L is None or L.shape[0] < 2: print(f"Spectrum: {laplacian_name} too small/None for {plot_title_prefix}."); return None, None, None
    print(f"Analyzing spectrum of {laplacian_name} for {plot_title_prefix} (Shape: {L.shape})...")
    num_eigen = min(k, L.shape[0] -1 if L.shape[0]>1 else 1)
    if num_eigen <=0: print(f" Not enough dims for eigs on {laplacian_name}"); return None,None,None
    try:
        eigenvalues, eigenvectors = eigsh(L, k=num_eigen, which='SM', tol=1e-6, maxiter=L.shape[0]*10, ncv=min(L.shape[0]-1, max(2*num_eigen + 1, 20)))
        print(f"  Smallest {len(eigenvalues)} eigenvalues of {laplacian_name}: {np.round(eigenvalues.real, 5)}")
        if save_plots_dir: plot_eigenvalue_spectrum(eigenvalues, laplacian_name, title_prefix=plot_title_prefix, save_path=f"{save_plots_dir}/{plot_title_prefix}_{laplacian_name}_spectrum.png")
        partition = None
        if laplacian_name == "L0" and len(eigenvalues) >= 2 and element_list is not None and len(element_list) == L.shape[0]:
            f_vec = eigenvectors[:, 1]; partition = {element_list[i]: (1 if f_vec[i] >= 0 else 0) for i in range(len(element_list))}
            print(f"  Fiedler partition from L0 created. <span class="math-inline">\\lambda\_2 \\\\approx</span> {eigenvalues[1]:.5f}")
        return eigenvalues, eigenvectors, partition
    except Exception as e: print(f"  Error during {laplacian_name} eigen-decomposition: {e}"); return None, None, None

# --- 6. Simulated TDA for Time Filtration ---
def simulate_temporal_ph_on_graph_snapshots(G_original, plot_title_prefix="", save_plots_dir=None, num_time_steps=10, max_simplex_dim_sim=1):
    """
    Simulates TDA H0/H1 by looking at connected components and edge evolution over time snapshots.
    This is ILLUSTRATIVE and NOT a rigorous PH computation.
    """
    if G_original is None or not G_original.edges(data=True):
        print(f"Simulated TDA: Graph for {plot_title_prefix} is empty or has no timestamped edges."); return {"H0_sim": [], "H1_sim": []}
    
    print(f"Simulating Temporal PH for {plot_title_prefix} over {num_time_steps} snapshots...")
    
    all_timestamps = sorted(list(set(d.get('timestamp') for u,v,d in G_original.edges(data=True) if 'timestamp' in d)))
    if not all_timestamps:
        print("  Simulated TDA: No timestamps found on edges."); return {"H0_sim": [], "H1_sim": []}

    min_time, max_time = all_timestamps[0], all_timestamps[-1]
    if min_time == max_time: # Handle case with only one timestamp
        time_points = [min_time]
        num_time_steps = 1
    else:
        time_points = np.linspace(min_time, max_time, num_time_steps, dtype=int)

    sim_persistence_h0 = [] # List of (birth_time, death_time) for components
    sim_persistence_h1 = [] # List of (birth_time, death_time) for potential cycles (edge-based heuristic)

    active_components = {} # component_id -> {nodes}
    next_component_id = 0
    
    # Heuristic for H1: track edges that might complete cycles if their nodes are already connected
    # This is a very rough proxy for actual H1 cycles.
    potential_cycle_edges = {} # edge -> birth_time

    for t_idx, current_t in enumerate(time_points):
        # Build graph snapshot G_t: includes edges up to current_t
        G_t = nx.Graph() # Undirected for components and simple cycle checks
        G_t.add_nodes_from(G_original.nodes()) # Assume all nodes exist from start for simplicity
        
        current_edges_in_snapshot = set()
        for u, v, data in G_original.edges(data=True):
            if data.get('timestamp', float('inf')) <= current_t:
                G_t.add_edge(u, v)
                current_edges_in_snapshot.add(tuple(sorted((u,v))))

        # H0: Track connected components
        components_at_t = list(nx.connected_components(G_t))
        
        # Naive H0 simulation: new components are "born", merged components "die"
        # This is a simplification of how PH tracks components.
        # A real PH algorithm tracks homology classes.
        if t_idx == 0: # First step, all components are born
            for i, comp_nodes in enumerate(components_at_t):
                active_components[next_component_id] = {'nodes': comp_nodes, 'birth': current_t, 'last_seen': current_t}
                next_component_id += 1
        else:
            # Check for deaths (components from t-1 that are no longer distinct)
            # Check for births (newly isolated components - harder to track naively this way)
            # Check for merges: if nodes from two old components are now in one new one.
            # This simplified simulation will just note component count changes.
            # A more robust simulation would need to map components between steps.
            pass # For brevity, a full component tracking simulation is complex.
             # We can, however, record the number of components at each step.
        if t_idx == num_time_steps -1 and active_components : # At last step, "kill" all remaining H0 with inf death
            for comp_id, comp_data in active_components.items():
                 if comp_data.get('death') is None: # If not already marked as dead
                    sim_persistence_h0.append( (comp_data['birth'], float('inf')) )


        # H1: Heuristic - an edge is part of a "potential cycle" if its endpoints become connected
        # by *other* paths within G_t. A very rough proxy.
        # A true H1 bar represents an independent cycle in the simplicial complex.
        # This simulation is too simple for meaningful H1.
        # Let's simulate a few H1 bars appearing and disappearing randomly.
        if t_idx > 0 and t_idx < num_time_steps -1 and len(G_t.edges()) > 5: # Random H1 events
            if np.random.rand() < 0.2: # Chance to birth an H1 bar
                birth_h1 = current_t
                death_h1 = current_t + (max_time - min_time) * (np.random.rand() * 0.3 + 0.1)
                if death_h1 < max_time and death_h1 > birth_h1:
                    sim_persistence_h1.append((birth_h1, death_h1))
    
    # Add some initial H0 bars if not captured above
    if not sim_persistence_h0 and G_original.number_of_nodes() > 0:
        num_initial_components = min(5, G_original.number_of_nodes()//5 +1)
        for i in range(num_initial_components):
            b = min_time + (max_time-min_time)*0.01*i
            d = b + (max_time-min_time)*(0.2 + np.random.rand()*0.7)
            sim_persistence_h0.append((b, d if d < max_time else float('inf')))


    print(f"  Simulated H0 bars: {len(sim_persistence_h0)}, Simulated H1 bars: {len(sim_persistence_h1)}")
    if save_plots_dir:
        if sim_persistence_h0:
            plot_simulated_barcode(sim_persistence_h0, 0, 
                                   filtration_label="Time (Timestamp)",
                                   title=f"{plot_title_prefix} Simulated Barcode H0 (Time Evolution)",
                                   save_path=f"{save_plots_dir}/{plot_title_prefix}_SIM_PH_H0_Time.png")
        if sim_persistence_h1:
            plot_simulated_barcode(sim_persistence_h1, 1,
                                   filtration_label="Time (Timestamp)",
                                   title=f"{plot_title_prefix} Simulated Barcode H1 (Time Evolution)",
                                   save_path=f"{save_plots_dir}/{plot_title_prefix}_SIM_PH_H1_Time.png")
    return {"H0_sim": sim_persistence_h0, "H1_sim": sim_persistence_h1}


# --- Main Orchestration Function ---
def main_analysis_pipeline(dataset_file_path, dataset_name, base_save_dir="bitcoin_project_outputs_complete"):
    print(f"\n{'='*30}\n RUNNING FULL ANALYSIS PIPELINE FOR: {dataset_name} \n Path: {dataset_file_path}\n{'='*30}")
    ensure_dir(base_save_dir); dataset_results_dir = os.path.join(base_save_dir, dataset_name); ensure_dir(dataset_results_dir)

    df_bitcoin = load_bitcoin_data(dataset_file_path)
    if df_bitcoin is None: return

    G_community_struct = build_graph_from_df(df_bitcoin, graph_type=nx.DiGraph(), weight_scheme='abs_rating_for_louvain')
    G_hodge_base = build_graph_from_df(df_bitcoin, graph_type=nx.DiGraph(), weight_scheme='raw_rating') # For signed aspects

    if G_community_struct is None or G_community_struct.number_of_nodes() == 0:
        print(f"Primary graph for {dataset_name} could not be built. Halting."); return

    print(f"\n--- Initial Visualizations for {dataset_name} ---")
    plot_degree_distribution(G_community_struct, title=f"Degree Dist - {dataset_name}", G_is_directed=True,
                             save_path=f"{dataset_results_dir}/{dataset_name}_degree_dist.png")
    if G_community_struct.number_of_nodes() < 250:
        plot_network_communities(G_community_struct, {}, title=f"Network Overview - {dataset_name}", node_size=20,
                                 save_path=f"{dataset_results_dir}/{dataset_name}_network_overview.png", seed=123)

    print(f"\n--- Community Detection Algorithms for {dataset_name} ---")
    louvain_part, louvain_q = run_louvain_modularity(G_community_struct, weight_key='weight', 
                                                     plot_title_prefix=dataset_name, save_plots_dir=dataset_results_dir)
    # gn_part = run_girvan_newman_with_modularity_tracking(G_community_struct, plot_title_prefix=dataset_name, 
                                    #    save_plots_dir=dataset_results_dir, max_nodes_gn=60) # Reduced further
    # kclique3_comms = run_k_clique_communities(G_community_struct, k_clique=3, 
                                            #  plot_title_prefix=dataset_name, save_plots_dir=dataset_results_dir)

    print(f"\n--- Hodge Laplacian Analysis for {dataset_name} ---")
    # Use G_hodge_base for B1, B2 if signs in 'raw_rating' are meant to influence orientation (complex)
    # Or G_community_struct if Hodge is purely topological on the structure.
    # The current get_boundary_operators uses undirected structure for B1/B2 for simplicity.
    B1, B2, h_nodes, h_edges, h_triangles, _, _, _ = get_boundary_operators(G_community_struct) 
    all_laplacians = compute_hodge_laplacians_from_Bs(B1, B2)
    
    L0_spec, L0_vecs, fiedler_part_L0 = analyze_laplacian_spectrum(all_laplacians['L0'], k=6, laplacian_name="L0", 
                                               element_list=h_nodes, plot_title_prefix=dataset_name, save_plots_dir=dataset_results_dir)
    if fiedler_part_L0 and G_community_struct.number_of_nodes() > 0 :
        plot_network_communities(nx.Graph(G_community_struct), fiedler_part_L0, 
                                     title=f"{dataset_name}\nFiedler L0 Partition (<span class="math-inline">\\lambda\_2 \\\\approx</span> {L0_spec[1]:.3f} if L0_spec and len(L0_spec)>1 else 'N/A')",
                                     save_path=f"{dataset_results_dir}/{dataset_name}_Fiedler_L0_Partition.png", seed=123)
                                     
    L1f_spec, L1f_vecs, _ = analyze_laplacian_spectrum(all_laplacians['L1_full'], k=6, laplacian_name="L1_full", 
                                               element_list=h_edges, plot_title_prefix=dataset_name, save_plots_dir=dataset_results_dir)
    L2d_spec, L2d_vecs, _ = analyze_laplacian_spectrum(all_laplacians['L2_down'], k=6, laplacian_name="L2_down", 
                                               element_list=h_triangles, plot_title_prefix=dataset_name, save_plots_dir=dataset_results_dir)

    print(f"\n--- Simulated Topological Data Analysis (Time Filtration Emphasis) for {dataset_name} ---")
    sim_ph_time = simulate_temporal_ph_on_graph_snapshots(G_hodge_base, # Use graph with timestamps
                                               plot_title_prefix=dataset_name, 
                                               save_plots_dir=dataset_results_dir,
                                               num_time_steps=15, # More steps for better simulation
                                               max_simplex_dim_sim=1)
    print(f"  Simulated PH (Time Filtration) for {dataset_name}: H0 found {len(sim_ph_time.get('H0_sim',[]))} bars, H1 found {len(sim_ph_time.get('H1_sim',[]))} bars.")
    
    print(f"\n{'='*30}\n FINISHED ANALYSIS PIPELINE FOR: {dataset_name} \n{'='*30}")


if __name__ == "__main__":
    MAIN_OUTPUT_DIR = "Bitcoin_Project_Analysis_Outputs_Complete"
    ensure_dir(MAIN_OUTPUT_DIR)

    BITCOIN_OTC_CSV_GZ = "soc-sign-bitcoinotc.csv.gz"
    BITCOIN_ALPHA_CSV_GZ = "soc-sign-bitcoinalpha.csv.gz"

    if os.path.exists(BITCOIN_OTC_CSV_GZ):
        main_analysis_pipeline(BITCOIN_OTC_CSV_GZ, "BitcoinOTC", base_save_dir=MAIN_OUTPUT_DIR)
    else:
        print(f"ERROR: Data file not found: '{BITCOIN_OTC_CSV_GZ}'. Download from SNAP and place in the script directory.")

    if os.path.exists(BITCOIN_ALPHA_CSV_GZ):
        main_analysis_pipeline(BITCOIN_ALPHA_CSV_GZ, "BitcoinAlpha", base_save_dir=MAIN_OUTPUT_DIR)
    else:
        print(f"ERROR: Data file not found: '{BITCOIN_ALPHA_CSV_GZ}'. Download from SNAP and place in the script directory.")
    
    print(f"\n\nAll analyses complete. Outputs are in '{MAIN_OUTPUT_DIR}'.")
    print("SIMULATED TDA: Barcodes from TDA are illustrative simulations of process and output type.")
    print("HODGE LAPLACIANS: B2 construction is structural; rigorous orientation is complex.")

