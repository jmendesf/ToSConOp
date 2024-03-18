import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.collections import LineCollection
import colorsys
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
import higra as hg
import numpy as np
from sklearn import datasets
import torch as tc

COLORS  = np.array(['#377eb8', '#ff7f00', '#4daf4a', '#a65628', '#f781bf', '#984ea3', '#999999', '#e41a1c', '#dede00'])
MARKERS = np.array(['o', '^', 's', 'X'])

def lighten_color(color_list, amount=0.25):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """    
    out = []
    for color in color_list:
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        lc = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
        out.append(lc)
    return 

def plot_graph(graph, X, y, idx=None): 
    
    sources, targets = graph.edge_list()
    segments = np.stack((X[sources, :], X[targets, :]), axis=1)
    lc = LineCollection(segments, zorder=0, colors='k')
    lc.set_linewidths(1)
    ax = plt.gca()
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlim(segments[:,:,0].min(), segments[:,:,0].max())
    ax.set_ylim(segments[:,:,1].min(), segments[:,:,1].max())
    ax.add_collection(lc)
    #plt.axis('equal')
    ec = COLORS[y%len(COLORS)]
    plt.scatter(X[:, 0], X[:, 1], s=30, c=ec, edgecolors='k', alpha=0.9)
    #ax.scatter(X[:, 0], X[:, 1], s=20, c='w', edgecolors='k')
    if idx is not None:
        iec = COLORS[y[idx]%len(COLORS)]
        plt.scatter(X[idx,0], X[idx,1], s=80, color=iec, marker='s', edgecolors='k')

def plot_dendrogram(tree, altitudes, n_clusters=0, lastp=5):
    linkage_matrix = hg.binary_hierarchy_to_scipy_linkage_matrix(tree, altitudes)
    extra = {} if lastp is None else dict(truncate_mode='lastp', p=lastp)
    set_link_color_palette(list(COLORS))
    dsort = np.sort(linkage_matrix[:,2]) 
    dendrogram(linkage_matrix, no_labels=True, above_threshold_color="k", color_threshold = dsort[-n_clusters+1], **extra)
    plt.yticks([])
    

def plot_clustering(X, y, idx=None):
    ec = COLORS[y%len(COLORS)]
    plt.scatter(X[:, 0], X[:, 1], s=15, linewidths=1.5, c=lighten_color(ec), edgecolors=ec, alpha=0.9)
    #plt.axis([X[:,0].min(), X[:,0].max(), X[:,1].min(), X[:,1].max()])
    plt.xticks(())
    plt.yticks(())
    if idx is not None:
        iec = COLORS[y[idx]%len(COLORS)]
        plt.scatter(X[idx,0], X[idx,1], s=30, color=iec, marker='s', edgecolors='k')

def create_dataset(n_samples=200, n_labeled=20, generator="varied"):
    
    
    def make_varied(n_samples, n_labeled):
        X, y = datasets.make_blobs(n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=170)#170
        return X, y
    
    def make_circles(n_samples, n_labeled):
        X, y = datasets.make_circles(n_samples, factor=.5, noise=.05, random_state=10) 
        return X, y

    def make_moons(n_samples, n_labeled):
        X, y = datasets.make_moons(n_samples, noise=.05, random_state=42)
        X, y = np.concatenate((X, X + (2.5, 0))), np.concatenate((y, y+2))
        return X, y

    def make_blobs(n_samples, n_labeled):
        X, y = datasets.make_blobs(n_samples, random_state=42)
        return X, y

    def make_aniso(n_samples, n_labeled):
        X, y = datasets.make_blobs(n_samples, random_state=170)
        X    = np.dot(X, [[0.6, -0.6], [-0.4, 0.8]])
        return X, y
    
    generators ={
        "varied": make_varied,
        "circles": make_circles,
        "moons": make_moons,
        "blobs": make_blobs,
        "aniso": make_aniso
    }
    
    
    
    X, y= generators[generator](n_samples, n_labeled)
    idx = np.arange(X.shape[0])
    graph, edge_weights = hg.make_graph_from_points(X,"knn+mst", n_neighbors=9)
    np.random.seed(42)
    np.random.shuffle(idx)
    data = {
        "X": X, 
        "y": y, 
        "n_clusters": len(np.unique(y)), 
        "labeled": idx[:n_labeled],
        "unlabeled": idx[n_labeled:],
        "graph": graph,
        "edge_weights": edge_weights,
    }
    return data

class OptimizerBPT:
    def __init__(self, graph, loss, hier_function, lr, optimizer="adam"):
        """
        Create an Optimizer utility object

        loss: function that takes a single torch tensor which support requires_grad = True and returns a torch scalar
        lr: learning rate
        optimizer: "adam" or "sgd"
        project: projection function for projected gradient descent optimization
        """
        self.graph = graph
        self.loss_function = loss
        self.hier_function = hier_function
        self.project = tc.relu
        self.history = []
        self.optimizer = optimizer
        self.lr = lr
        self.best = None
        self.best_loss = float("inf")

    def fit(self, data, iter=1000, debug=False, min_lr=None):
        """
        Fit the given data

        data: torch tensor, input data
        iter: int, maximum number of iterations
        debug: int, if > 0, print current loss value and learning rate every debug iterations
        min_lr: float, minimum learning rate (an LR scheduler is used), if None, no LR scheduler is used 
        """
        data = data.clone().requires_grad_(True)
        if self.optimizer == "adam":
            optimizer = tc.optim.Adam([data], lr=self.lr, amsgrad=True)
        else:
            optimizer = tc.optim.SGD([data], lr=self.lr)

        if min_lr:
            lr_scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10)

        for t in range(iter):
            optimizer.zero_grad()

            if self.project:
                data_proj = self.project(data) 
            else:
                data_proj = data

            tree, altitudes = self.hier_function(self.graph, data_proj)
            loss = self.loss_function(tree, altitudes)
            loss.backward()

            optimizer.step()  
            loss_value = loss.item()
            
            self.history.append(loss_value) 
            if loss_value < self.best_loss:
                self.best_loss = loss_value
                self.best = data_proj.clone()
                
            if min_lr:
                lr_scheduler.step(loss_value)
                if optimizer.param_groups[0]['lr'] <= min_lr:
                    break

            if debug and t % debug == 0:
                print("Iteration {}: Loss: {:.4f}, LR: {}".format(t, loss_value, optimizer.param_groups[0]['lr']))
        t, a = bpt_canonical(self.graph, self.best)
        return t, a.detach().numpy()

    def show_history(self):
        """
        Plot loss history
        """
        plt.plot(self.history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss history")
        #plt.show()

def imshow(image, click_event=None, cmap=None, figsize=None, vmin=None, vmax=None):
    """
    Show an image at true scale
    """
    import matplotlib.pyplot as plt
    dpi = 80
    margin = 0.5  # (5% of the width/height of the figure...)
    if figsize is None:
        h, w = image.shape[:2]
    else:
        h, w = figsize

        
    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * w / dpi, (1 + margin) * h / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(image, interpolation='none', cmap=cmap, vmin=vmin, vmax=vmax)
    
    plt.axis('off')
    plt.show()
    
    return fig, ax

def make_pairs(labels, idx):
    """
    Create all pairs from the provided labeld set. This function returns:
    :param labels: provided labels
    :param idx: vertex indices of the provided labels
    :return: a tuple ``pairs=(vertices1, vertices2)`` indexing all the pairs of elements in ``ìdx``
    """
    pairs   = labels[None] == labels[:,None]
    src,dst = np.triu_indices(pairs.shape[0], 1)
    labels  = pairs[src,dst]
    src, dst   = idx[src], idx[dst]
    
    pos = src[labels], dst[labels]
    neg = src[np.logical_not(labels)], dst[np.logical_not(labels)]
    return pos, neg

def print_max_tree(tree, altitudes, attribute=None):
    if attribute is None:
        attribute = altitudes
        
    hg.print_partition_tree(tree, altitudes=np.max(altitudes) - altitudes, attribute=attribute)
    
def imshow(image, click_event=None, cmap=None, figsize=None, vmin=None, vmax=None):
    """
    Show an image at true scale
    """
    import matplotlib.pyplot as plt
    dpi = 80
    margin = 0.5  # (5% of the width/height of the figure...)
    if figsize is None:
        h, w = image.shape[:2]
    else:
        h, w = figsize

        
    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * w / dpi, (1 + margin) * h / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(image, interpolation='none', cmap=cmap, vmin=vmin, vmax=vmax)
    
    plt.axis('off')
    plt.show()
    
    return fig, ax

class Optimizer:
    def __init__(self, loss, lr, optimizer="adam"):
        """
        Create an Optimizer utility object

        loss: function that takes a single torch tensor which support requires_grad = True and returns a torch scalar
        lr: learning rate
        optimizer: "adam" or "sgd"
        project: projection function for projected gradient descent optimization
        """
        self.loss_function = loss
        self.project = lambda image: tc.clamp(image, min=0, max=1)
        self.history = []
        self.optimizer = optimizer
        self.lr = lr
        self.best = None
        self.best_loss = float("inf")

    def fit(self, data, iter=1000, debug=False, min_lr=None):
        """
        Fit the given data

        data: torch tensor, input data
        iter: int, maximum number of iterations
        debug: int, if > 0, print current loss value and learning rate every debug iterations
        min_lr: float, minimum learning rate (an LR scheduler is used), if None, no LR scheduler is used 
        """
        data = data.clone().requires_grad_(True)
        if self.optimizer == "adam":
            optimizer = tc.optim.Adam([data], lr=self.lr, amsgrad=True)
        else:
            optimizer = tc.optim.SGD([data], lr=self.lr)

        if min_lr:
            lr_scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10)

        for t in range(iter):
            optimizer.zero_grad()

            if self.project:
                data_proj = self.project(data) 
            else:
                data_proj = data

            loss = self.loss_function(data_proj)
            loss.backward()

            optimizer.step()  
            loss_value = loss.item()
            
            self.history.append(loss_value) 
            if loss_value < self.best_loss:
                self.best_loss = loss_value
                self.best = data_proj.clone()
                
            if min_lr:
                lr_scheduler.step(loss_value)
                if optimizer.param_groups[0]['lr'] <= min_lr:
                    break

            if debug and t % debug == 0:
                print("Iteration {}: Loss: {:.4f}, LR: {}".format(t, loss_value, optimizer.param_groups[0]['lr']))
        return self.best

    def show_history(self):
        """
        Plot loss history
        """
        plt.plot(self.history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss history")
        plt.show()
        
def attribute_depth(tree, altitudes):
    """
    Compute the depth of any node of the tree which is equal to the largest altitude 
    in the subtree rooted in the current node. 

    :param tree: input tree
    :param altitudes: np array (1d), altitudes of the input tree nodes
    :return: np array (1d), depth of the tree nodes
    """
    return hg.accumulate_sequential(tree, altitudes[:tree.num_leaves()], hg.Accumulators.max)

def attribute_saddle_nodes(tree, attribute):
    """
    Let n be a node and let an be an ancestor of n. The node an has a single child node that contains n denoted by ch(an -> n). 
    The saddle and base nodes associated to a node n for the given attribute values are respectively the closest ancestor an  
    of n and the node ch(an -> n) such that there exists a child c of an with attr(ch(an -> n)) < attr(c). 

    :param tree: input tree
    :param attribute: np array (1d), attribute of the input tree nodes
    :return: (np array, np array), saddle and base nodes of the input tree nodes for the given attribute
    """
    max_child_index = hg.accumulate_parallel(tree, attribute, hg.Accumulators.argmax)
    child_index = hg.attribute_child_number(tree)
    main_branch = child_index == max_child_index[tree.parents()]
    main_branch[:tree.num_leaves()] = True

    saddle_nodes = hg.propagate_sequential(tree, np.arange(tree.num_vertices(), dtype=np.int64)[tree.parents()], main_branch)
    base_nodes = hg.propagate_sequential(tree, np.arange(tree.num_vertices(), dtype=np.int64), main_branch)
    return saddle_nodes, base_nodes