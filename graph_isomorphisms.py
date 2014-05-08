import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import brewer2mpl


# This helper function will be used to generate the seed for the randomness, so that we can be consistent
import hashlib
def hash_string(text):
    return int(hashlib.md5(text).hexdigest()[0:7], 16)


def setup_plots():
	# set some nicer defaults for matplotlib
	from matplotlib import rcParams
	rcParams['figure.figsize'] = (14, 6)
	rcParams['figure.dpi'] = 150
	rcParams['patch.linewidth'] = 0.7
	rcParams['patch.edgecolor'] = '#262626'
	rcParams['axes.edgecolor'] = '#262626'
	rcParams['xtick.color'] = '#262626'
	rcParams['ytick.color'] = '#262626'
	rcParams['text.color'] = '#262626'
	rcParams['axes.titlesize'] = 20

def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecessary plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()

def visualize_beacons(G, beacons):
    labels = {}
    node_colors=[]
    node_sizes=[]
    for i,node in enumerate(G.nodes()):
        node_colors.append('#65abd0')
        node_sizes.append(600)
        if node in beacons :
            labels[node] = node
        else:
            labels[node] = ''
            node_colors[i] = '#A0CBE2'
            node_sizes[i] = 100
    return (labels, node_colors, node_sizes)

def noisy_remove(G, p, seed=None):
	from numpy.random import RandomState
	prng = RandomState(seed)
	G_copy = G.copy()
	for (a,b) in G_copy.edges():
		if prng.rand() < p and len(G_copy[a])>1 and len(G_copy[b])>1:
			G_copy.remove_edges_from([(a,b)])
	return G_copy

def plot_graphs(G1, G2, beacons_G1, beacons_G2):
	plt.subplot(121)
	plt.axis('off')
	plt.title('$G_1$')
	(labels, node_colors, node_sizes) = visualize_beacons(G1, beacons_G1)
	pos = nx.spring_layout(G1, weight=None)
	nx.draw_networkx_nodes(G1, pos, node_color=node_colors, node_size=node_sizes, font_size=18)
	nx.draw_networkx_labels(G1, pos, font_size=17, labels=labels, font_color = '#262626')
	nx.draw_networkx_edges(G1, pos, width=2, alpha=0.3)

	plt.subplot(122)
	plt.axis('off')
	plt.title('$G_2$')
	(labels, node_colors, node_sizes) = visualize_beacons(G2, beacons_G2)
	pos = nx.spring_layout(G2, weight=None)
	nx.draw_networkx_nodes(G2, pos, node_color=node_colors, node_size=node_sizes, font_size=18)
	nx.draw_networkx_labels(G2, pos, font_size=17, labels=labels, font_color = '#262626')
	nx.draw_networkx_edges(G2, pos, width=2, alpha=0.3)


def isomap_project(p):
	from sklearn.manifold import Isomap
	from sklearn.metrics.pairwise import euclidean_distances
	pairwise_distances = euclidean_distances(p)
	# Add noise to the similarities
	n_samples = p.shape[0]
	noise = np.random.rand(n_samples, n_samples) * 0
	noise = noise + noise.T
	noise[np.arange(noise.shape[0]), np.arange(noise.shape[0])] = 0
	pairwise_distances += noise
	return Isomap(n_neighbors=10, n_components=2 ).fit_transform(pairwise_distances)




from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans as KMeansGood
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.datasets.samples_generator import make_blobs

class KMeans(BaseEstimator):
 
    def __init__(self, k, max_iter=100, random_state=0, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
 
    def _e_step(self, X):
        self.labels_ = euclidean_distances(X, self.cluster_centers_,
                                     squared=True).argmin(axis=1)
 
    def _average(self, X):
        return X.mean(axis=0)
 
    def _m_step(self, X):
        X_center = None
        for center_id in range(self.k):
            center_mask = self.labels_ == center_id
            if not np.any(center_mask):
                # The centroid of empty clusters is set to the center of
                # everything
                if X_center is None:
                    X_center = self._average(X)
                self.cluster_centers_[center_id] = X_center
            else:
                self.cluster_centers_[center_id] = \
                    self._average(X[center_mask])
 
    def fit(self, X, y=None):
        n_samples = X.shape[0]
        vdata = np.mean(np.var(X, 0))
 
        random_state = check_random_state(self.random_state)
        self.labels_ = random_state.permutation(n_samples)[:self.k]
        self.cluster_centers_ = X[self.labels_]
 
        for i in xrange(self.max_iter):
            centers_old = self.cluster_centers_.copy()
 
            self._e_step(X)
            self._m_step(X)
 
            if np.sum((centers_old - self.cluster_centers_) ** 2) < self.tol * vdata:
                break
 
        return self
 
class KMedians(KMeans):
 
    def _e_step(self, X):
        self.labels_ = manhattan_distances(X, self.cluster_centers_).argmin(axis=1)
 
    def _average(self, X):
        return np.median(X, axis=0)

def nearest_neigbor(node, candidates):
	from sklearn.neighbors import NearestNeighbors
	nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(candidates) 
	distances, indices = nbrs.kneighbors(node)
	return distances, indices

def cluster_colors(G, beacons, kmeans_labels, colors):
    j=0
    clustered_colors=[]
    for i,node in enumerate(G.nodes()):
        if node not in beacons:
            clustered_colors.append(colors[kmeans_labels[j]])
            j = j + 1
        else:
            clustered_colors.append((101./256,171./256,208./256))
    return clustered_colors

def paint_clusters(G1, G2, beacons_G1, beacons_G2, kmeans_labels1, kmeans_labels2):
	accent_colors = brewer2mpl.get_map('Accent', 'qualitative',8).mpl_colors
	plt.subplot(121)
	plt.axis('off')
	plt.title('$G_1$')
	(labels, node_colors, node_sizes) = visualize_beacons(G1, beacons_G1)
	clustered_colors = cluster_colors(G1, beacons_G1, kmeans_labels1, accent_colors)

	pos = nx.spring_layout(G1, weight=None)
	nx.draw_networkx_nodes(G1, pos, node_color=clustered_colors, node_size=node_sizes, font_size=18)
	nx.draw_networkx_labels(G1, pos, font_size=17, labels=labels, font_color = '#262626')
	nx.draw_networkx_edges(G1, pos, width=2, alpha=0.3)


	plt.subplot(122)
	plt.axis('off')
	plt.title('$G_2$')
	(labels, node_colors, node_sizes) = visualize_beacons(G2, beacons_G2)
	clustered_colors = cluster_colors(G2, beacons_G2, kmeans_labels2, accent_colors)

	pos2_init= {key:value for (key, value) in zip(beacons_G2,[pos[beacon] for beacon in beacons_G1])}
	pos2 = nx.spring_layout(G2, weight=None, pos = pos2_init)

	nx.draw_networkx_nodes(G2, pos=pos2, node_color=clustered_colors, node_size=node_sizes, font_size=18)
	nx.draw_networkx_labels(G2, pos=pos2, font_size=17, labels=labels, font_color = '#262626')
	nx.draw_networkx_edges(G2, pos=pos2, width=2, alpha=0.3)

def find_beacons_sample_inverse(G, num_of_beacons=3, seed=None):
    # Sample a beacon based on its degree
	from numpy.random import RandomState
	prng = RandomState(seed)	
	degrees = np.array(nx.degree(G).values())
	return prng.choice(np.arange(len(degrees)), num_of_beacons, p=(1./degrees) *1./sum(1./degrees), replace=False )

def shortest_path_project(G, beacons):
    projection = np.zeros((G.number_of_nodes() - len(beacons), len(beacons)))
    
    for i,beacon in enumerate(beacons):
        lengths = nx.shortest_path_length(G, source=beacon)
        node_index = 0
        for node in lengths:
            if node in beacons:
                continue
            projection[node_index][i] = lengths[node]
            node_index += 1           
    return projection

def effective_resistance_project(G, beacons):
    from numpy.linalg import pinv
    projection = np.zeros((G.number_of_nodes() - len(beacons), len(beacons)))
    L = nx.laplacian_matrix(G)
    B = nx.incidence_matrix(G).T
    B_e = B.copy()
    
    L_pseudo = pinv(L)
    for i in xrange(B.shape[0]):
        min_ace = np.min(np.where(B[i,:] ==1)[1])
        B_e[i, min_ace] = -1
    
    for i,beacon in enumerate(beacons):
        node_index = 0
        for j,node in enumerate(G.nodes()):
            if node in beacons:
                continue
                
            battery = np.zeros((B_e.shape[1],1))
            battery[i] = 1
            battery[node_index] = -1

            p = L_pseudo * battery
            projection[node_index][i] = abs(p[i] - p[j])
            node_index += 1 
    return projection

def count_matches(G1, G2, beacons_G1, beacons_G2, best, anonymous_mapping):
	row_labels1 = [i for i in G1.nodes() if i not in beacons_G1]
	row_labels2 = [i for i in G2.nodes() if i not in beacons_G2]
	num_points = len(G1.nodes())
	matching_matrix = np.zeros((num_points,num_points))
	atleast_one = False
	correct = 0
	for node in xrange(len(row_labels1)):
		i = best[node]
		if row_labels2[i] == anonymous_mapping[node]:
			matching_matrix[row_labels2[i]][anonymous_mapping[node]] = 1
			atleast_one = True
			correct += 1
		else:
			matching_matrix[row_labels2[i]][anonymous_mapping[node]] = -1
	for node in beacons_G1:
		if atleast_one:
			matching_matrix[node][node] = 0.4
		else:
			matching_matrix[node][node] = 1
	return (correct, matching_matrix)


def match(G1, G2, beacon_percentage, electrical=False):
	beacons_G1 = find_beacons_sample_inverse(G1,int(beacon_percentage * G1.number_of_nodes()))
	beacons_G2 = beacons_G1
	rest_of_nodes = [x for x in G1.nodes() if x not in beacons_G1]

	anonymous_mapping = dict(zip(beacons_G1,beacons_G1))
	anonymous_mapping.update(zip(rest_of_nodes,np.random.permutation(rest_of_nodes)))
	G2 = nx.relabel_nodes(G2,anonymous_mapping, copy=True)

	if electrical:
		p1 = effective_resistance_project(G1,beacons_G1)
		p2 = effective_resistance_project(G2,beacons_G2)
	else:
		p1 = shortest_path_project(G1,beacons_G1)
		p2 = shortest_path_project(G2,beacons_G2)

	d= euclidean_distances(p1,p2)
	best = {k:v for k,v in find_best_match(d)}
	return (best, anonymous_mapping, beacons_G1, beacons_G2)

def find_best_match(pairwise_distances):
    d = np.copy(pairwise_distances)
    matchings=[]
    rows = np.array(np.arange(d.shape[0]))
    cols = np.array(np.arange(d.shape[1]))
    for i in range(d.shape[0]-1):
        s = np.argsort(d, axis=None)
        index = np.unravel_index(s[0],d.shape)
        matchings.append((rows[index[0]], cols[index[1]]))
        d = np.delete(d, index[0],0)
        rows = np.delete(rows, index[0])
        d = np.delete(d, index[1],1)
        cols = np.delete(cols, index[1])
    matchings.append((rows[0], cols[0]))
    return matchings

def print_matching(G, matching_matrix):
	centrality = nx.degree_centrality(G)
	c2 = sorted(centrality, key=centrality.get)
	fig, ax = plt.subplots(1)
	p = ax.pcolormesh(matching_matrix[c2,:][:,c2],cmap=brewer2mpl.get_map('PRGn', 'Diverging',5).mpl_colormap, clim=(-46,46))
	fig.colorbar(p)


