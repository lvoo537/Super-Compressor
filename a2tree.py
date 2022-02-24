
"""
Assignment 2: Quadtree Compression


=== Module Description ===
This module contains classes implementing the quadtree.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional
from copy import deepcopy


# No other imports allowed

def combine(matrix: List[List], bl: List[List] , br : List[List]) -> List[List]:
    if len(bl) > len(br):
        u = 0
        while u < len(br):
            matrix.append(bl[u] + br[u])
            u += 1
        while u < len(bl):
            matrix.append(bl[u])
            u += 1
    elif len(br) > len(bl):
        j = 0
        while j < len(bl):
            matrix.append(bl[j] + br[j])
            j += 1
        while j <len(br):
            matrix.append(br[j])
            j += 1
    elif len(br) == len(bl):
        for z in range(len(br)):
            matrix.append(bl[z] + br[z])

def mean_and_count(matrix: List[List[int]]) -> Tuple[float, int]:
    """
    Returns the average of the values in a 2D list
    Also returns the number of values in the list
    """
    total = 0
    count = 0
    for row in matrix:
        for v in row:
            total += v
            count += 1
    return total / count, count


def standard_deviation_and_mean(matrix: List[List[int]]) -> Tuple[float, float]:
    """
    Return the standard deviation and mean of the values in <matrix>

    https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Note that the returned average is a float.
    It may need to be rounded to int when used.
    """
    avg, count = mean_and_count(matrix)
    total_square_error = 0
    for row in matrix:
        for v in row:
            total_square_error += ((v - avg) ** 2)
    return math.sqrt(total_square_error / count), avg


class QuadTreeNode:
    """
    Base class for a node in a quad tree
    """

    def __init__(self) -> None:
        pass

    def tree_size(self) -> int:
        raise NotImplementedError

    def convert_to_pixels(self, width: int, height: int) -> List[List[int]]:
        raise NotImplementedError

    def preorder(self) -> str:
        raise NotImplementedError


class QuadTreeNodeEmpty(QuadTreeNode):
    """
    An empty node represents an area with no pixels included
    """

    def __init__(self) -> None:
        super().__init__()

    def tree_size(self) -> int:
        """
        Note: An empty node still counts as 1 node in the quad tree
        """
        # TODO: implement this method
        return 1

    def convert_to_pixels(self, width: int, height: int) -> List[List[int]]:
        """
        Convert to a properly formatted empty list
        >>> sample_leaf = QuadTreeNodeEmpty()
        >>> sample_leaf.convert_to_pixels(0, 0)
        []
        """
        # Note: Normally, this method should return an empty list or a list of
        # empty lists. However, when the tree is mirrored, this returned list
        # might not be empty and may contain the value 255 in it. This will
        # cause the decompressed image to have unexpected white pixels.
        # You may ignore this caveat for the purpose of this assignment.
        return [[255] * width for _ in range(height)]

    def preorder(self) -> str:
        """
        The letter E represents an empty node
        """
        return 'E'

    def _swap(self):
        return QuadTreeNodeEmpty()


class QuadTreeNodeLeaf(QuadTreeNode):
    """
    A leaf node in the quad tree could be a single pixel or an area in which
    all pixels have the same colour (indicated by self.value).
    """

    value: int  # the colour value of the node

    def __init__(self, value: int) -> None:
        super().__init__()
        assert isinstance(value, int)
        self.value = value

    def tree_size(self) -> int:
        """
        Return the size of the subtree rooted at this node
        """
        # TODO: complete this method
        return 1

    def convert_to_pixels(self, width: int, height: int) -> List[List[int]]:
        """
        Return the pixels represented by this node as a 2D list

        >>> sample_leaf = QuadTreeNodeLeaf(5)
        >>> sample_leaf.convert_to_pixels(2, 2)
        [[5, 5], [5, 5]]
        >>> sample_leaf.convert_to_pixels(1, 1)
        [[5]]
        >>> sample_leaf.convert_to_pixels(1, 3)
        [[5], [5], [5]]
        >>> sample_leaf.convert_to_pixels(2, 3)
        [[5, 5], [5, 5], [5, 5]]
        >>> sample_leaf.convert_to_pixels(4, 3)
        [[5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5]]
        >>> sample_leaf.convert_to_pixels(4, 2)
        [[5, 5, 5, 5], [5, 5, 5, 5]]
        """
        # TODO: complete this method
        outer_list = []
        for i in range(height):
            outer_list.append([self.value] * width)
        return outer_list

    def preorder(self) -> str:
        """
        A leaf node is represented by an integer value in the preorder string
        """
        return str(self.value)

    def _swap(self):
        return QuadTreeNodeLeaf(self.value)


class QuadTreeNodeInternal(QuadTreeNode):
    """
    An internal node is a non-leaf node, which represents an area that will be
    further divided into quadrants (self.children).

    The four quadrants must be ordered in the following way in self.children:
    bottom-left, bottom-right, top-left, top-right

    (List indices increase from left to right, bottom to top)

    Representation Invariant:
    - len(self.children) == 4
    """
    children: List[Optional[QuadTreeNode]]

    def __init__(self) -> None:
        """
        Order of children: bottom-left, bottom-right, top-left, top-right
        """
        super().__init__()

        # Length of self.children must be always 4.
        self.children = [None, None, None, None]

    def tree_size(self) -> int:
        """
        The size of the subtree rooted at this node.

        This method returns the number of nodes that are in this subtree,
        including the root node.
        >>> t = QuadTree(0)
        >>> t.build_quad_tree([[1, 2], [3, 4]])
        >>> t.tree_size()
        5
        >>> t2 = QuadTree(0)
        >>> t2.build_quad_tree([[1,2,3],[4,5,6], [7,8,9]])
        >>> t2.tree_size()
        17
        >>> t3 = QuadTree(0)
        >>> t3.build_quad_tree([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
        >>> t3.tree_size()
        21
        """
        # TODO: complete this method
        size_of_others = 0
        for i in range(len(self.children)):
            size_of_others += self.children[i].tree_size()
        return size_of_others + 1

    def pixels(self) -> List[List]:
        lst = []
        for i in range(len(self.children)):
            if isinstance(self.children[i], QuadTreeNodeLeaf):
                lst.append(self.children[i].value)
            elif isinstance(self.children[i], QuadTreeNodeInternal):
                lst.append(self.children[i].pixels)
        return [lst]

    def convert_to_pixels(self, width: int, height: int) -> List[List[int]]:
        """
        Return the pixels represented by this node as a 2D list.

        You'll need to recursively get the pixels for the quadrants and
        combine them together.

        Make sure you get the sizes (width/height) of the quadrants correct!
        Read the docstring for split_quadrants() for more info.
        >>> t = QuadTree(0)
        >>> t.build_quad_tree([[1,2,3],[4,5,6],[7,8,9]])
        >>> t.convert_to_pixels()
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> t1 = QuadTree(0)
        >>> t1.build_quad_tree([[1, 2, 2], [4, 15, 15]])
        >>> t1.convert_to_pixels()
        [[1, 2, 2], [4, 15, 15]]
        >>> t2 = QuadTree(0)
        >>> t2.build_quad_tree([[1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 8, 9, 10, 22], [7, 8, 9, 10, 11, 12, 13],[14, 15, 16, 17, 18, 19, 20]])
        >>> t2.convert_to_pixels()
        [[1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 8, 9, 10, 22], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20]]
        >>> t3 = QuadTree(0)
        >>> t3.build_quad_tree([[1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 8, 9, 10, 22], [7, 8, 9, 10, 11, 12, 13],[14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]])
        >>> t3.convert_to_pixels()
        [[1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 8, 9, 10, 22], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
        >>> t4 = QuadTree(0)
        >>> t4.build_quad_tree([[1, 2, 3, 4, 5, 6], [4, 5, 6, 8, 9, 10], [7, 8, 9, 10, 11, 12], [14, 15, 16, 17, 18, 19]])
        >>> t4.convert_to_pixels()
        [[1, 2, 3, 4, 5, 6], [4, 5, 6, 8, 9, 10], [7, 8, 9, 10, 11, 12], [14, 15, 16, 17, 18, 19]]
        >>> t5 = QuadTree(0)
        >>> t5.build_quad_tree([[1, 2, 3, 4, 5, 6], [4, 5, 6, 8, 9, 10], [7, 8, 9, 10, 11, 12],[14, 15, 16, 17, 18, 19], [21, 22, 23, 24, 25, 26]])
        >>> t5.convert_to_pixels()
        [[1, 2, 3, 4, 5, 6], [4, 5, 6, 8, 9, 10], [7, 8, 9, 10, 11, 12], [14, 15, 16, 17, 18, 19], [21, 22, 23, 24, 25, 26]]
        >>> t6 = QuadTree(0)
        >>> t6.build_quad_tree([[1,2,3,4]])
        >>> t6.convert_to_pixels()
        [[1, 2, 3, 4]]
        >>> t7 = QuadTree(0)
        >>> t7.build_quad_tree([[1], [2], [3]])
        >>> t7.convert_to_pixels()
        [[1], [2], [3]]
        >>> t8 = QuadTree(0)
        >>> t8.build_quad_tree([[1,2,3]])
        >>> t8.convert_to_pixels()
        [[1, 2, 3]]
        >>> t7 = QuadTree(0)
        >>> t7.build_quad_tree([[1], [2], [3], [4]])
        >>> t7.convert_to_pixels()
        [[1], [2], [3], [4]]
        """
        # TODO: complete this method
        matrix = []
        hsplit = height // 2
        vsplit = width // 2
        bl_width = vsplit
        bl_height = hsplit
        br_width = width - vsplit
        br_height = hsplit
        tl_width = vsplit
        tl_height = height - hsplit
        tr_width = width - vsplit
        tr_height = height - hsplit
        for i in range(len(self.children)):
            if i == 0:
                bl = self.children[i].convert_to_pixels(bl_width, bl_height)
            elif i == 1:
                br = self.children[i].convert_to_pixels(br_width, br_height)
            elif i == 2:
                tl = self.children[i].convert_to_pixels(tl_width, tl_height)
            elif i == 3:
                tr = self.children[i].convert_to_pixels(tr_width, tr_height)
        combine(matrix, bl, br)
        combine(matrix, tl, tr)
        return matrix

    def preorder(self) -> str:
        """
        Return a string representing the preorder traversal or the tree rooted
        at this node. See the docstring of the preorder() method in the
        QuadTree class for more details.

        An internal node is represented by an empty string in the preorder
        string.
        >>> t = QuadTree(0)
        >>> t.build_quad_tree([[1, 2], [3, 4]])
        >>> t.preorder()
        ',1,2,3,4'
        >>> t2 = QuadTree(0)
        >>> t2.build_quad_tree([[1,2,3],[4,5,6], [7,8,9]])
        >>> t2.preorder()
        ',1,,E,E,2,3,,E,4,E,7,,5,6,8,9'
        >>> t3 = QuadTree(0)
        >>> t3.build_quad_tree([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
        >>> t3.preorder()
        ',,E,1,E,4,,2,3,5,6,,E,7,E,10,,8,9,11,12'
        """
        # TODO: complete this method
        str1 = ""
        for i in range(len(self.children)):
            str1 += "," + self.children[i].preorder()
        return str1

    def restore_from_preorder(self, lst: List[str], start: int) -> int:
        """
        Restore subtree from preorder list <lst>, starting at index <start>
        Return the number of entries used in the list to restore this subtree
        >>> lst1 = ',,E,1,E,4,,2,3,5,6,,E,7,E,10,,8,9,11,12'.split(',')
        >>> internal1 = QuadTreeNodeInternal()
        >>> internal1.restore_from_preorder(lst1, 0)
        21
        >>> lst2 = ',1,,,1,2,3,,1,2,3,4,1,E,,1,2,3,4,E,,E,E,E,,1,2,3,,1,2,E,4'.split(',')
        >>> internal2 = QuadTreeNodeInternal()
        >>> internal2.restore_from_preorder(lst2,0)
        33
        """

        # This assert will help you find errors.
        # Since this is an internal node, the first entry to restore should
        # be an empty string
        assert lst[start] == ''
        # TODO: complete this method
        element = start + 1
        i = 0
        while i < 4:
            if lst[element] == '':
                internal2 = QuadTreeNodeInternal()
                self.children[i] = internal2
                element = internal2.restore_from_preorder(lst, element)
                i += 1
            elif lst[element] == 'E':
                self.children[i] = QuadTreeNodeEmpty()
                element += 1
                i += 1
            else:
                self.children[i] = QuadTreeNodeLeaf(int(lst[element]))
                element += 1
                i += 1
        return element

    def mirror(self) -> None:
        """
        Mirror the bottom half of the image represented by this tree over
        the top half

        Example:
            Original Image
            1 2
            3 4

            Mirrored Image
            3 4 (this row is flipped upside down)
            3 4

        See the assignment handout for a visual example.
        >>> p = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> q = QuadTree()
        >>> q.build_quad_tree(p, True)
        >>> q.convert_to_pixels()
        [[1, 2, 3], [1, 2, 3], [1, 255, 255]]
        """
        # TODO
        self.children[2] = deepcopy(self.children[0])
        self.children[3] = deepcopy(self.children[1])
        self.children[2]._swap()
        self.children[3]._swap()

    def _swap(self):
        self.children[0], self.children[2] = self.children[2], self.children[0]
        self.children[1], self.children[3] = self.children[3], self.children[1]
        for i in range(len(self.children)):
            self.children[i]._swap()


class QuadTree:
    """
    The class for the overall quadtree
    """

    loss_level: float
    height: int
    width: int
    root: Optional[QuadTreeNode]  # safe to assume root is an internal node

    def __init__(self, loss_level: int = 0) -> None:
        """
        Precondition: the size of <pixels> is at least 1x1
        """
        self.loss_level = float(loss_level)
        self.height = -1
        self.width = -1
        self.root = None

    def build_quad_tree(self, pixels: List[List[int]],
                        mirror: bool = False) -> None:
        """
        Build a quad tree representing all pixels in <pixels>
        and assign its root to self.root

        <mirror> indicates whether the compressed image should be mirrored.
        See the assignment handout for examples of how mirroring works.
        """
        # print('building_quad_tree...')
        self.height = len(pixels)
        self.width = len(pixels[0])
        self.root = self._build_tree_helper(pixels)
        if mirror:
            self.root.mirror()
        return

    def _build_tree_helper(self, pixels: List[List[int]]) -> QuadTreeNode:
        """
        Build a quad tree representing all pixels in <pixels>
        and return the root

        Note that self.loss_level should affect the building of the tree.
        This method is where the compression happens.

        IMPORTANT: the condition for compressing a quadrant is the standard
        deviation being __LESS THAN OR EQUAL TO__ the loss level. You must
        implement this condition exactly; otherwise, you could fail some
        test cases unexpectedly.
        """
        # TODO: complete this method
        if pixels == [] or pixels[0] == []:
            empty = QuadTreeNodeEmpty()
            return empty
        elif len(pixels) == 1 and len(pixels[0]) == 1:
            leaf = QuadTreeNodeLeaf(pixels[0][0])
            return leaf
        else:
            quadrants = self._split_quadrants(pixels)
            internal = QuadTreeNodeInternal()
            for i in range(len(quadrants)):
                if quadrants[i] == []:
                    internal2 = self._build_tree_helper(quadrants[i])
                    internal.children[i] = internal2
                elif quadrants[i] != []:
                    std_and_mean = standard_deviation_and_mean(quadrants[i])
                    if std_and_mean[0] <= self.loss_level:
                        mean1 = round(std_and_mean[1])
                        leaf = QuadTreeNodeLeaf(mean1)
                        internal.children[i] = leaf
                    else:
                        internal2 = self._build_tree_helper(quadrants[i])
                        internal.children[i] = internal2
        return internal

    @staticmethod
    def _split_quadrants(pixels: List[List[int]]) -> List[List[List[int]]]:
        """
        Precondition: size of <pixels> is at least 1x1
        Returns a list of four lists of lists, correspoding to the quadrants in
        the following order: bottom-left, bottom-right, top-left, top-right

        IMPORTANT: when dividing an odd number of entries, the smaller half
        must be the left half or the bottom half, i.e., the half with lower
        indices.

        Postcondition: the size of the returned list must be 4
        >>> example = QuadTree(0)
        >>> example._split_quadrants([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        [[[1]], [[2, 3]], [[4], [7]], [[5, 6], [8, 9]]]
        >>> example2 = QuadTree(0)
        >>> example2._split_quadrants([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13,14,15,16]])
        [[[1, 2], [5, 6]], [[3, 4], [7, 8]], [[9, 10], [13, 14]], [[11, 12], [15, 16]]]
        >>> example3 = QuadTree(0)
        >>> example3._split_quadrants([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
        [[[1], [4]], [[2, 3], [5, 6]], [[7], [10], [13]], [[8, 9], [11, 12], [14, 15]]]
        >>> example4 = QuadTree(0)
        >>> example4._split_quadrants([[1,2],[3,4]])
        [[[1]], [[2]], [[3]], [[4]]]
        >>> example5 = QuadTree(0)
        >>> example5._split_quadrants([[1,2,3,4],[5,6, 7,8],[9,10,11,12], [13,14,15,16]])
        [[[1, 2], [5, 6]], [[3, 4], [7, 8]], [[9, 10], [13, 14]], [[11, 12], [15, 16]]]
        >>> example6 = QuadTree(0)
        >>> example6._split_quadrants([[1], [2], [3], [4]])
        [[], [[1], [2]], [], [[3], [4]]]
        >>> example7 = QuadTree(0)
        >>> example7._split_quadrants([[1], [2], [3], [4], [5]])
        [[], [[1], [2]], [], [[3], [4], [5]]]
        >>> example8 = QuadTree(0)
        >>> example8._split_quadrants([[1,2]])
        [[], [], [[1]], [[2]]]
        >>> example9 = QuadTree(0)
        >>> example9._split_quadrants([[1,2,3]])
        [[], [], [[1]], [[2, 3]]]
        >>> example10 = QuadTree(0)
        >>> example10._split_quadrants([[1,2,3,4]])
        [[], [], [[1, 2]], [[3, 4]]]
        >>> example11 = QuadTree(0)
        >>> example11._split_quadrants([[1,2,3,4,5]])
        [[], [], [[1, 2]], [[3, 4, 5]]]
        """
        # TODO: complete this method
        bottom_left = []
        bottom_right = []
        top_left = []
        top_right = []
        hsplit = len(pixels) // 2
        vsplit = len(pixels[0]) // 2
        if len(pixels) == 1:
            if len(pixels[0]) != 1:
                top_left.append(pixels[0][:vsplit])
                top_right.append(pixels[0][vsplit:])
        else:
            if len(pixels[0]) == 1:
                for i in range(len(pixels)):
                    if i < hsplit:
                        bottom_right.extend([pixels[i]])
                    elif i >= hsplit:
                        top_right.extend([pixels[i]])
            elif len(pixels[0]) != 1:
                for i in range(len(pixels)):
                    lst = []
                    for j in range(len(pixels[0])):
                        lst.append(pixels[i][j])
                    if i < hsplit:
                        bottom_left.extend([lst[:vsplit]])
                        bottom_right.extend([lst[vsplit:]])
                    elif i >= hsplit:
                        top_left.extend([lst[:vsplit]])
                        top_right.extend([lst[vsplit:]])
        return [bottom_left, bottom_right, top_left, top_right]

    def tree_size(self) -> int:
        """
        Return the number of nodes in the tree, including all Empty, Leaf, and
        Internal nodes.
        """
        return self.root.tree_size()

    def convert_to_pixels(self) -> List[List[int]]:
        """
        Return the pixels represented by this tree as a 2D matrix
        """
        return self.root.convert_to_pixels(self.width, self.height)

    def preorder(self) -> str:
        """
        return a string representing the preorder traversal of the quadtree.
        The string is a series of entries separated by comma (,).
        Each entry could be one of the following:
        - empty string '': represents a QuadTreeNodeInternal
        - string of an integer value such as '5': represents a QuadTreeNodeLeaf
        - string 'E': represents a QuadTreeNodeEmpty

        For example, consider the following tree with a root and its 4 children
                __      Root       __
              /      |       |        \
            Empty  Leaf(5), Leaf(8), Empty

        preorder() of this tree should return exactly this string: ",E,5,8,E"

        (Note the empty-string entry before the first comma)
        """
        return self.root.preorder()

    @staticmethod
    def restore_from_preorder(lst: List[str],
                              width: int, height: int) -> QuadTree:
        """
        Restore the quad tree from the preorder list <lst>
        The preorder list <lst> is the preorder string split by comma

        Precondition: the root of the tree must be an internal node (non-leaf)
        """
        tree = QuadTree()
        tree.width = width
        tree.height = height
        tree.root = QuadTreeNodeInternal()
        tree.root.restore_from_preorder(lst, 0)
        return tree


def maximum_loss(original: QuadTreeNode, compressed: QuadTreeNode) -> float:
    """
    Given an uncompressed image as a quad tree and the compressed version,
    return the maximum loss across all compressed quadrants.

    Precondition: original.tree_size() >= compressed.tree_size()

    Note: original, compressed are the root nodes (QuadTreeNode) of the
    trees, *not* QuadTree objects

    >>> pixels = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> orig, comp = QuadTree(0), QuadTree(2)
    >>> orig.build_quad_tree(pixels)
    >>> comp.build_quad_tree(pixels)
    >>> maximum_loss(orig.root, comp.root)
    1.5811388300841898
    >>> pixels = [[1, 2, 3, 4], [5, 6,7,8], [9,10,11,12]]
    >>> orig, comp = QuadTree(0), QuadTree(4)
    >>> orig.build_quad_tree(pixels)
    >>> comp.build_quad_tree(pixels)
    >>> maximum_loss(orig.root, comp.root)
     3.452052529534663
    """
    # TODO: complete this function
    max_loss = 0
    if isinstance(compressed, QuadTreeNodeInternal) and isinstance(original, QuadTreeNodeInternal):
        for i in range(len(compressed.children)):
            if isinstance(compressed.children[i],
                          QuadTreeNodeLeaf) and isinstance(original.children[i],
                                                           QuadTreeNodeInternal):
                std_and_mean = standard_deviation_and_mean(original.children[i].pixels())[0]
                if std_and_mean >= max_loss:
                    max_loss = std_and_mean
            elif isinstance(compressed.children[i],
                            QuadTreeNodeInternal) and isinstance(
                    original.children[i], QuadTreeNodeInternal):
                max_loss2 = maximum_loss(original.children[i],
                                         compressed.children[i])
                if max_loss2 >= max_loss:
                    max_loss = max_loss2
        return max_loss


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    # import python_ta
    # python_ta.check_all()
