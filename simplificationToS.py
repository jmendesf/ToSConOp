import random
from helper import *
from PIL import Image


def split_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def bound(low, high, value):
    if value > high:
        return high
    if value < low:
        return low
    return value


class Node:
    def __init__(self, name, alt, parent):
        self.name = name
        self.alt = alt
        self.parent = parent
        self.children = []
        self.proper_part = []

        # Bounds of the node, i.e. the "closest" smaller/higher altitude of its parent/children
        self.lower_bound = None
        self.upper_bound = None

        # Children nodes corresponding to bounds (M+/M-)
        self.children_lower_bound = None
        self.children_upper_bound = None

        self.removed = False

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return not self.children

    # Change the node's altitude. Depending on the node's configuration,
    # this operation can lead to fusing with the parent (hence removal of the node) and/or the children nodes.
    # Returns:
    # If the node is not removed by the altitude change, returns itself
    # Else return its parent
    def change_node_altitude_in_bounds(self, new_alt):
        if self.alt == new_alt:
            return self
        parent_alt = self.__get_parent_alt()
        self.__compute_bounds()

        # Bounding of the new altitude
        # -1 for lower/upper bounds represents infinite
        # new_alt must remain between the lower and upper bounds of the node
        new_alt = self.__bound_value(new_alt)

        # if the new altitude is strictly between the lower and upper bound of the node,
        # there is no impact on the relations of the node
        if self.__is_value_strictly_between_bounds(new_alt):
            self.alt = new_alt
            return self

        # Look for impacted children (i.e. M+ / M-)
        if new_alt == self.lower_bound:
            self.__add_lower_bound_children()
            impacted_children = self.children_lower_bound
        else:
            self.__add_upper_bound_children()
            impacted_children = self.children_upper_bound

        # If the new altitude of the self node reaches some of its children,
        # these children fuse with the self node and are removed.
        # The children of the fused nodes are now children of the self node
        for impacted_child in impacted_children:
            impacted_child.__fuse_to_parent()

        self.alt = new_alt

        # if the new altitude of the self node reaches its parent node, the parent absorbs the self node,
        # and each child of the self node is now a child of the parent node
        if new_alt == parent_alt:
            parent = self.parent
            self.__fuse_to_parent()
            return parent

        return self

    # Returns the altitude closest to the node between the upper and lower bound
    def closest_bound(self):
        self.__compute_bounds()
        if self.parent is None:
            print(self)
        if self.lower_bound == -1 or self.upper_bound == self.alt:
            return self.upper_bound
        if self.upper_bound == -1:
            return self.lower_bound

        distance_lower = self.alt - self.lower_bound
        distance_upper = self.upper_bound - self.alt

        return self.upper_bound if distance_upper > distance_lower else self.lower_bound

    # The size of the proper part, i.e. the nb of pixels in the set
    def get_nb_proper_part(self):
        return len(self.proper_part)

    # m
    def distance_to_parent(self):
        par_alt = self.__get_parent_alt()
        return par_alt - self.alt if par_alt > self.alt else self.alt - par_alt

    # nb of gray levels separating the node from its closest node
    def distance_to_closest(self):
        self.__compute_bounds()

        dist_up = self.upper_bound - self.alt
        dist_low = self.alt - self.lower_bound

        if self.lower_bound == -1:
            return dist_up
        if self.upper_bound == -1:
            return dist_low

        return dist_up if dist_up <= dist_low else dist_low

    # Returns the mean value between all the neighbours (parent/children) of the node
    # weighted: do not use, keep False
    def get_mean_neighboring_value(self, weighted=False):
        if self.parent is not None:
            if weighted:
                alts = [(self.__get_parent_alt(), self.parent.get_nb_proper_part())]
            else:
                alts = [(self.__get_parent_alt(), 1)]
        for child in self.children:
            if weighted:
                alts.append((child.alt, child.get_nb_proper_part()))
            else:
                alts.append((child.alt, 1))
        sum_alt = 0
        weight = 0
        for a, w in alts:
            sum_alt += a * w
            weight += w

        return int(round(sum_alt / weight, 0))

    def __fuse_to_parent(self):
        # for each child, remove the relation to self and link child to parent
        for child in self.children:
            child.parent = self.parent
            self.parent.children.append(child)
        self.children.clear()

        # proper part is given to parent
        self.parent.proper_part += self.proper_part
        self.proper_part.clear()

        # relation to parent is removed
        self.parent.children.remove(self)
        self.parent = None
        self.removed = True

    # Compute the bounds of the node
    # Bounds are defined depending on the neighbouring nodes' (parent and children) altitudes
    def __compute_bounds(self):
        parent_alt = self.__get_parent_alt()
        self.upper_bound = None
        self.lower_bound = None
        for child in self.children:
            if child.alt > self.alt:
                if self.upper_bound is None or child.alt < self.upper_bound:
                    self.upper_bound = child.alt
            elif child.alt < self.alt:
                if self.lower_bound is None or child.alt > self.lower_bound:
                    self.lower_bound = child.alt

        if parent_alt > self.alt:
            if self.upper_bound is None or self.upper_bound > parent_alt:
                self.upper_bound = parent_alt
        elif parent_alt < self.alt:
            if self.lower_bound is None or self.lower_bound < parent_alt:
                self.lower_bound = parent_alt

        # if no finite upper/lower bound was found, -1 represents an infinite bound
        if self.upper_bound is None:
            self.upper_bound = -1
        # we consider that the lower bound cannot be lower than 0 for now
        if self.lower_bound is None:
            if self.alt == 0:
                self.lower_bound = 0
            else:
                self.lower_bound = -1

    # add children node corresponding to the lower bound
    def __add_lower_bound_children(self):
        self.children_lower_bound = []
        if self.lower_bound == -1:
            return
        for child in self.children:
            if child.alt == self.lower_bound:
                self.children_lower_bound.append(child)

    # add children node corresponding to the upper bound
    def __add_upper_bound_children(self):
        self.children_upper_bound = []
        if self.upper_bound == -1:
            return
        for child in self.children:
            if child.alt == self.upper_bound:
                self.children_upper_bound.append(child)

    def __bound_value(self, value):
        if value < self.lower_bound != -1:
            return self.lower_bound
        elif value > self.upper_bound != -1:
            return self.upper_bound
        if self.is_root():
            if value <= self.alt:
                return self.alt
            else:
                return self.upper_bound
        return value

    def get_lower_bound(self):
        self.__compute_bounds()
        return self.lower_bound

    def get_upper_bound(self):
        self.__compute_bounds()
        return self.upper_bound

    def __is_value_strictly_between_bounds(self, value):
        return ((value > self.alt and (value < self.upper_bound or self.upper_bound == -1)) or
                (value < self.alt and (value > self.lower_bound or self.lower_bound == -1)))

    # Returns the parent's altitude. If the node is the root, returns the altitude of the node
    def __get_parent_alt(self):
        if self.parent is None:
            return self.alt
        else:
            return self.parent.alt

    # debugging
    def print_bounding(self):
        self.__compute_bounds()
        print("Bounding:", str(self.lower_bound), "<", str(self.alt), "<", self.upper_bound)

    def __str__(self):
        return str(self.name) + "-" + str(self.alt) + " area: " + str(len(self.proper_part))


def compute_children_map(sources, targets):
    children = {}
    i = 0
    for child in sources:
        if children.get(targets[i]) is None:
            children[targets[i]] = [child]
        else:
            children[targets[i]].append(child)
        i += 1
    return children


class TreeOfShapes:
    def __init__(self, image):
        self.image = image
        self.nodes = {}
        self.__hg_tab_to_obj(image)

    def __hg_tab_to_obj(self, image):
        nb_pixels = image.shape[0] * image.shape[1]
        g, altitude = hg.component_tree_tree_of_shapes_image2d(image)

        # map of nodes
        self.nb_nodes = len(g.parents()) - nb_pixels
        # node altitudes
        self.altitude = altitude

        sources, targets = g.edge_list()
        children_map = compute_children_map(sources[nb_pixels:], targets[nb_pixels:])
        root = g.root()

        altitude_root = (
            int(bound(0, 255, round(altitude[root] * 255, 0)))) if 0 < altitude[root] < 1 \
            else int(altitude[root])
        self.root = Node(root, altitude_root, None)
        self.nodes[root] = self.root

        # construction of the tree structure from the root
        node_list = [self.root]
        while node_list:
            node = node_list.pop(0)
            children = children_map.get(node.name)

            if children is not None:
                for child in children:
                    altitude_child = (
                        int(bound(0, 255, round(altitude[child] * 255, 0)))) if 0 < altitude[child] < 1 \
                        else int(altitude[child])
                    child_node = Node(child, altitude_child, node)
                    self.nodes[child_node.name] = child_node
                    node.children.append(child_node)
                    node_list.append(child_node)

        self.compute_proper_parts(nb_pixels, g.parents())

    def compute_proper_parts(self, nb_pix_im, parents):
        pix_index = 0
        for pixel in parents[:nb_pix_im]:
            self.nodes[pixel].proper_part.append(pix_index)
            pix_index += 1

    # Change the alt of the given node name
    # the new alt is bounded by the neighbourhood of the node
    # Impacts can fusion with parent/children
    def change_alt_of_node(self, node_name, new_alt):
        node = self.nodes.get(node_name)
        if node is not None:
            while node.alt != new_alt:
                node = node.change_node_altitude_in_bounds(new_alt)
            return node
        else:
            print("Node", node_name, "not found")
            return None

    # Simplifies the tree of shapes from the leaves to the root, by discarding the nodes that have a proper part smaller
    # than area_value
    # Discarding policy depends on to_parent flag:
    # True: node is merged to parent
    # False: node is merged to its "closest" node (can be parent or child).
    def filter_tree_proper_part_bottom_up(self, proper_part_value, to_parent=False):
        self.__filter_tree_proper_part_bottom_up(proper_part_value, self.root, to_parent)

    # Consecutive proper part filtering, from starting value to end value with a step.
    def filter_tree_proper_part_bottom_up_consecutive(self, starting_pp_value, step, end_value, to_parent=True):
        while starting_pp_value < end_value:
            self.filter_tree_proper_part_bottom_up(starting_pp_value, to_parent)
            starting_pp_value += step

    def __filter_tree_proper_part_bottom_up(self, proper_part_value, node, to_parent):
        childlist = [] + node.children
        for child in childlist:
            self.__filter_tree_proper_part_bottom_up(proper_part_value, child, to_parent)
        if not node.is_root() and not node.removed:
            if node.get_nb_proper_part() < proper_part_value:
                if not to_parent:
                    self.change_alt_of_node(node.name, node.closest_bound())
                else:
                    self.change_alt_of_node(node.name, node.parent.alt)

    # fuses the node to its closer relative(s) if its area (= nb pixels of the proper part) is lesser than area_value
    # going from the root to the leaves
    def filter_tree_area(self, proper_part_value):
        nodelist = [self.root]
        visited = {}
        while nodelist:
            node = nodelist.pop(0)
            while not node.is_root() and node.get_nb_proper_part() < proper_part_value:
                node = self.change_alt_of_node(node.name, node.parent.alt)
            for child in node.children:
                if visited.get(child.name) is None:
                    visited[child.name] = True
                    nodelist.append(child)

    # Assigns to every node in the tree its mean neighbouring value
    def filter_tree_mean(self, weighted=False):
        nodelist = [self.root]
        visited = {}
        while nodelist:
            node = nodelist.pop(0)
            if not node.is_root():
                node = self.change_alt_of_node(node.name, node.get_mean_neighboring_value(weighted))
            for child in node.children:
                if visited.get(child.name) is None:
                    visited[child.name] = True
                    nodelist.append(child)

    # Simplify the tree by removing nodes that are further away than a certain value from their parent
    # If pp_value != 0: a criterion is also the proper part of the node
    def filter_tree_distance_to_parent(self, distance_value, pp_value=0):
        treat_area = pp_value != 0
        nodelist = [self.root]
        visited = {}
        while nodelist:
            node = nodelist.pop(0)
            while not node.is_root() and node.distance_to_parent() > distance_value:
                if treat_area and node.get_nb_proper_part() > pp_value:
                    break
                node = self.change_alt_of_node(node.name, node.parent.alt)
            for child in node.children:
                if visited.get(child.name) is None:
                    visited[child.name] = True
                    nodelist.append(child)

    # Filter nodes that have a distance to their closest bound greater than a certain treshold
    def filter_tree_gap(self, gap_value, area=0):
        treat_area = area != 0
        nodelist = [self.root]
        visited = {}
        while nodelist:
            node = nodelist.pop(0)
            # while not node.is_root() and node.distance_to_closest() < gap_value:
            while not node.is_root() and node.distance_to_parent() < gap_value:
                if treat_area and node.get_nb_proper_part() > area:
                    break
                node = self.change_alt_of_node(node.name, node.parent.alt)
            for child in node.children:
                if visited.get(child.name) is None:
                    visited[child.name] = True
                    nodelist.append(child)

    # Filter the tree by shifting node altitudes randomly within a certain interval
    # Default order: nodes sorted by ascending proper part
    # To parent: random distance will be forced to be towards the parent
    # To children: the random distance won't be towards the parent
    # Random pick: nodes chosen at random
    # reverse_sort: nodes of bigger proper part first
    def filter_random_shifts(self, stop_percent, random_interval, to_parent=False, to_children=False, random_pick=False,
                             reverse_sort=False):
        sorted_list = self.__get_nodes_sorted_proper_part()
        nb_nodes = len(sorted_list)
        target_value = int((stop_percent * nb_nodes) / 100)
        print("nb node is", nb_nodes, "and target value is", target_value)
        while len(sorted_list) > target_value:
            if(random_pick):
                i = random.randint(0, len(sorted_list) - 1)
                node = sorted_list.pop(i)
                while node.removed:
                    i = random.randint(0, len(sorted_list) - 1)
                    node = sorted_list.pop(i)
            else:
                if not reverse_sort:
                    node = sorted_list.pop(0)
                    while node.removed:
                        node = sorted_list.pop(0)
                else:
                    node = sorted_list.pop(len(sorted_list) - 1)
                    while node.removed:
                        node = sorted_list.pop(len(sorted_list) - 1)

            if node.is_root():
                continue

            interval_0 = node.alt - random_interval
            if interval_0 < 0:
                interval_0 = 0

            interval_1 = node.alt + random_interval
            if node.get_lower_bound() == -1:
                interval_0 = node.alt
            if node.get_upper_bound() == -1:
                interval_1 = node.alt

            if to_parent:
                if node.parent is not None:
                    if node.parent.alt > node.alt:
                        interval_0 = node.alt
                    else:
                        interval_1 = node.alt
            elif to_children:
                if len(node.children) > 0 and node.parent is not None:
                    if node.parent.alt > node.alt:
                        interval_1 = node.alt
                    else:
                        interval_0 = node.alt

            v = random.randint(interval_0, interval_1)
            node.change_node_altitude_in_bounds(v)
            if not node.removed:
                for index, n in enumerate(sorted_list):
                    if n.get_nb_proper_part() >= node.get_nb_proper_part():
                        sorted_list.insert(index, node)
                        break

    # Quantize the grayscale in nb_g_values values
    def filter_tree_quantization(self, nb_g_values):
        low_gv = 0
        step_gv = int(255 / nb_g_values)
        target_gv = step_gv
        gv_map = {}
        for i in range(0, 256):
            gv_map[i] = (low_gv, target_gv if target_gv <= 255 else 255)
            if i >= target_gv:
                low_gv += step_gv
                target_gv += step_gv

        sorted_list = self.__get_nodes_sorted_proper_part()
        sorted_list.reverse()
        while sorted_list:
            node = sorted_list.pop(0)
            while node.removed and sorted_list:
                node = sorted_list.pop(0)
            if not sorted_list:
                break
            if node.is_root():
                continue
            alt_range = gv_map[node.alt]
            target_alt = alt_range[0] if (node.alt - alt_range[0]) > (alt_range[1] - node.alt) else alt_range[1]
            self.change_alt_of_node(node.name, target_alt)

    # Consecutive area filtering
    def apply_consecutive_area_filters(self, starting_value, ending_value, increment_value):
        while starting_value < ending_value:
            self.filter_tree_area(starting_value)
            starting_value += increment_value

    # Reconstructs the image from the tree
    def reconstruct_image(self):
        arr1d = self.__get_1d_canvas()
        nodelist = [self.root]
        while nodelist:
            node = nodelist.pop(0)
            for px in node.proper_part:
                arr1d[px] = node.alt
            nodelist += node.children

        list_img = split_list(arr1d, self.image.shape[1])
        im_array = np.asarray(list_img, dtype=np.uint8)
        image = Image.fromarray(im_array)
        return image

    def get_node(self, node_name):
        return self.nodes.get(node_name)

    def get_nb_nodes(self):
        nodelist = [self.root]
        nb = 0
        while nodelist:
            node = nodelist.pop(0)
            nb += 1
            nodelist += node.children
        return nb

    def __get_nodes_sorted_proper_part(self):
        nodestack = [self.root]
        result = []
        while nodestack:
            node = nodestack.pop(0)
            result.append(node)
            nodestack += node.children
        result.sort(key=lambda x: x.get_nb_proper_part())
        return result

    def __get_1d_canvas(self):
        return [0] * self.image.shape[0] * self.image.shape[1]

    # debugging
    def print_tree(self):
        nodelist = [self.root]
        while nodelist:
            node = nodelist.pop(0)
            print(node, "children: ", end="")
            for child in node.children:
                print(child, end=", ")
            node.print_bounding()
            nodelist += node.children

    # returns a 1D image s.t. each pixel's value is the node to which this pixel is proper part
    def node_label_image(self):
        arr1d = self.__get_1d_canvas()
        nodelist = [self.root]
        while nodelist:
            node = nodelist.pop(0)
            for px in node.proper_part:
                arr1d[px] = node.name
            nodelist += node.children
        return arr1d

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        # Structural check
        label_map = {}
        label_img1 = self.node_label_image()
        label_img2 = other.node_label_image()

        if len(label_img1) != len(label_img2):
            print("false because length")
            return False

        for label1, label2 in zip(label_img1, label_img2):
            lab = label_map.get(label1)
            if lab is None:
                label_map[label1] = label2
            elif lab != label2:
                print("false because", lab, "is not", label2)
                return False

        # Node info check (altitude values)
        im1 = self.reconstruct_image()
        im2 = other.reconstruct_image()

        return im1 == im2
