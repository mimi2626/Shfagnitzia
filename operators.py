'''
This file is part of QuantifierComplexity.
'''
import math
from collections import namedtuple
from itertools import chain
import numpy as np

vectorized_isclose = np.vectorize(math.isclose)
vectorized_round = np.vectorize(round)

Operator = namedtuple("Operator", "name func input_types output_type")
SetPlaceholder = namedtuple("Operator", "name func input_types output_type")

Operator.__repr__ = lambda self: self.name
Operator.__str__ = lambda self: self.name

def index_func(idx: np.ndarray, set_repr: list, verbose=False) -> list:
    """Return set reprs of idx[i]'th object of given set for model i.

    The set with the idx[i]'th object of a given set, is a singleton.

    Args:
        idx: An array of the length of the model universe. Each entry
            contains an integer. The integer k position i signifies
            that the k'th object of model i (the i'th model in 
            the enumeration of models in the universe) should be
            returned as a singleton (or empty set if there is no k'th 
            object).
        set_repr: Contains binary numpy arrays of shape(model_size,
            nr_of_models_of_this_model_size), for model_size 0 to 
            max_model_size. Column i represents the elements 
            of a given set for model i (the i'th model in the 
            enumeration of models in the universe). A 1 at position j,i 
            means that the j'th object of model i is present in the set,  
            and a 0 means that that object is not present in the set.
        verbose: True or False. Print intermediate results for 
            debugging.

    Returns:
        A list with numpy arrays of shape(model_size,
            nr_of_models_of_this_model_size), for model_size 0 to 
            max_model_size. Each column i represents a set for a given 
            model. Each of these sets is a singleton, containing the 
            idx[n]'th object of the given set_repr for model n.

    """
    if verbose:
        print()
        print("i =", idx)
    idx_start_of_block = 0
    out = []
    for size_minus_one, set_repr_for_size in enumerate(set_repr):
        idx_end_of_block = idx_start_of_block + set_repr_for_size.shape[1]
        # The selection of the idx integer per model, 
        # for the models of the current size.
        idx_selection = idx[idx_start_of_block : idx_end_of_block]
        idx_start_of_block = idx_end_of_block
        result = np.zeros_like(set_repr_for_size)
        # Matrix that shows, per model (the columns), for each object in
        # the given set representation, the how manieth object that is.
        # When an object in the model is not in the set, it puts NaN.
        cum_set_repr = np.where(
            set_repr_for_size == 1, set_repr_for_size.cumsum(0), np.nan
        )
        # For model (column) i, select the idx_selection[i]'th object of 
        # the given set, by putting True at that position, and False 
        # otherwise.
        result = cum_set_repr == idx_selection
        out.append(result.astype(int))
        if verbose:
            print("idx_selection =", idx_selection)
            print("idx_selection_no_zero =", idx_selection)
            print("set_repr_for_size =", set_repr_for_size)
            print("cum_set_repr =", cum_set_repr)
            print("result =", result)
            print("out =", out)
    return out


def subset_func(set_repr_0: list, set_repr_1: list) -> np.array:
    """Compute, per model, whether set_0 subseteq set_1.

    Args:
        set_rep_0: Contains binary numpy arrays of shape(model_size,
            nr_of_models_of_this_model_size), for model_size 0 to 
            max_model_size. Column i represents the elements 
            of set_0 for model i (the i'th model in the enumeration
            of models in the universe). A 1 at position j,i means that 
            the j'th object of model i is present in the set, and a 0  
            means that that object is not present in the set.
        set_rep_1: Similarly to set_rep_0.

    Returns:
        A numpy array containing Booleans. 
        Boolean at index i represents the meaning of 
        (set_0 subseteq set_1), given the ith model in the universe.

    """
    out = np.array([])
    # set_0 and set_1 are numpy arrays of the same shape, representing
    # the meaning of set_0 and set_1 in all models of a particular 
    # size.
    for size, (set_0, set_1) in enumerate(zip(set_repr_0, set_repr_1)):
        subset = (
            # Apply along column axis.
            # Give False for each collum in which there is a 1
            # on a position in set_0 where there is a 0 on that position
            # in set B. Give True otherwise.
            np.apply_along_axis(max, 0, (set_0 & (1 - set_1))) == 0
        )
        out = np.append(out, subset)
    return out.astype(bool)

def identity_func(set_repr_0: list, set_repr_1: list) -> np.array:
    """Compute, per model, whether set_0 subseteq set_1.

    Args:
        set_rep_0: Contains binary numpy arrays of shape(model_size,
            nr_of_models_of_this_model_size), for model_size 0 to 
            max_model_size. Column i represents the elements 
            of set_0 for model i (the i'th model in the enumeration
            of models in the universe). A 1 at position j,i means that 
            the j'th object of model i is present in the set, and a 0  
            means that that object is not present in the set.
        set_rep_1: Similarly to set_rep_0.

    Returns:
        A numpy array containing Booleans. 
        Boolean at index i represents the meaning of 
        (set_0 subseteq set_1), given the ith model in the universe.

    """
    out = np.array([])
    # set_0 and set_1 are numpy arrays of the same shape, representing
    # the meaning of set_0 and set_1 in all models of a particular 
    # size.
    for size, (set_0, set_1) in enumerate(zip(set_repr_0, set_repr_1)):
        subset = (
            # Apply along column axis.
            # Give False for each collum in which there is a 1
            # on a position in set_0 where there is a 0 on that position
            # in set B. Give True otherwise.
            np.apply_along_axis(max, 0, ((set_0 & (1 - set_1)) | (set_1 & (1 - set_0)))) == 0
        )
        out = np.append(out, subset)
    return out.astype(bool)

def non_identity_func(set_repr_0: list, set_repr_1: list) -> np.array:
    """Compute, per model, whether set_0 subseteq set_1.

    Args:
        set_rep_0: Contains binary numpy arrays of shape(model_size,
            nr_of_models_of_this_model_size), for model_size 0 to 
            max_model_size. Column i represents the elements 
            of set_0 for model i (the i'th model in the enumeration
            of models in the universe). A 1 at position j,i means that 
            the j'th object of model i is present in the set, and a 0  
            means that that object is not present in the set.
        set_rep_1: Similarly to set_rep_0.

    Returns:
        A numpy array containing Booleans. 
        Boolean at index i represents the meaning of 
        (set_0 subseteq set_1), given the ith model in the universe.

    """
    out = np.array([])
    # set_0 and set_1 are numpy arrays of the same shape, representing
    # the meaning of set_0 and set_1 in all models of a particular 
    # size.
    for size, (set_0, set_1) in enumerate(zip(set_repr_0, set_repr_1)):
        subset = (
            # Apply along column axis.
            # Give False for each collum in which there is a 1
            # on a position in set_0 where there is a 0 on that position
            # in set B. Give True otherwise.
            np.apply_along_axis(max, 0, ((set_0 & (1 - set_1)) | (set_1 & (1 - set_0)))) == 1
        )
        out = np.append(out, subset)
    return out.astype(bool)

def entail_func(set_repr_0: list, set_repr_1: list) -> np.array:
    out = []
    # set_0 and set_1 are numpy arrays of the same shape, representing
    # the meaning of set_0 and set_1 in all models of a particular size.
    for size, (set_0, set_1) in enumerate(zip(set_repr_0, set_repr_1)):
        # Append a np.array of the same shape as set_0 and set_1.
        # For each position, put a 1, if set_0 has 0 or set_1 has 1,
        # on that position. Put a 0 otherwise.
        out.append(set_1 | (1 - set_0))
    return out

def opp_entail_func(set_repr_0: list, set_repr_1: list) -> np.array:
    out = []
    # set_0 and set_1 are numpy arrays of the same shape, representing
    # the meaning of set_0 and set_1 in all models of a particular size.
    for size, (set_0, set_1) in enumerate(zip(set_repr_0, set_repr_1)):
        # Append a np.array of the same shape as set_0 and set_1.
        # For each position, put a 1, if set_0 has 1 or set_1 has 0,
        # on that position. Put a 0 otherwise.
        out.append(set_0 | (1 - set_1))
    return out

def neg_entail_func(set_repr_0: list, set_repr_1: list) -> np.array:
    out = []
    # set_0 and set_1 are numpy arrays of the same shape, representing
    # the meaning of set_0 and set_1 in all models of a particular size.
    for size, (set_0, set_1) in enumerate(zip(set_repr_0, set_repr_1)):
        # Append a np.array of the same shape as set_0 and set_1.
        # For each position, put a 1, if set_0 has 1 and set_1 has 0,
        # on that position. Put a 0 otherwise.
        out.append(set_0 & (1 - set_1))
    return out

def neg_opp_entail_func(set_repr_0: list, set_repr_1: list) -> np.array:
    out = []
    # set_0 and set_1 are numpy arrays of the same shape, representing
    # the meaning of set_0 and set_1 in all models of a particular size.
    for size, (set_0, set_1) in enumerate(zip(set_repr_0, set_repr_1)):
        # Append a np.array of the same shape as set_0 and set_1.
        # For each position, put a 1, if set_0 has 0 and set_1 has 1,
        # on that position. Put a 0 otherwise.
        out.append(set_1 & (1 - set_0))
    return out

def iff_func(set_repr_0: list, set_repr_1: list) -> np.array:
    out = []
    # set_0 and set_1 are numpy arrays of the same shape, representing
    # the meaning of set_0 and set_1 in all models of a particular size.
    for size, (set_0, set_1) in enumerate(zip(set_repr_0, set_repr_1)):
        # Append a np.array of the same shape as set_0 and set_1.
        # For each position, put a 1, if set_0 has 0 and set_1 has 0,
        # or if both have 1
        # on that position. Put a 0 otherwise.
        out.append((set_0 & set_1) | ((1 - set_0) & (1 - set_1)))
    return out

def neg_iff_func(set_repr_0: list, set_repr_1: list) -> np.array:
    out = []
    # set_0 and set_1 are numpy arrays of the same shape, representing
    # the meaning of set_0 and set_1 in all models of a particular size.
    for size, (set_0, set_1) in enumerate(zip(set_repr_0, set_repr_1)):
        # Append a np.array of the same shape as set_0 and set_1.
        # For each position, put a 1, if set_0 has 0 or set_1 has 0,
        # and set_0 has 1 or set_1 has 1 (XOR)
        # on that position. Put a 0 otherwise.
        out.append((set_0 | set_1) & ((1 - set_0) | (1 - set_1)))
    return out





def diff_func(set_repr_0: list, set_repr_1: list) -> list:
    """Compute, per model, the set_repr of set_0 \ set_1.

    Args:
        set_rep_0: Contains binary numpy arrays of shape(model_size,
            nr_of_models_of_this_model_size), for model_size 0 to 
            max_model_size. Column i represents the elements 
            of set_0 for model i (the i'th model in the enumeration
            of models in the universe). A 1 at position j,i means that 
            the j'th object of model i is present in the set, and a 0  
            means that that object is not present in the set.
        set_rep_1: Similarly to set_rep_0.

    Returns:
        A list with numpy arrays of shape(model_size,
            nr_of_models_of_this_model_size), for model_size 0 to 
            max_model_size. Column i represents the elements 
            of set 0 \ set 1 for model i (in the enumeration of models 
            by generate_universe).

    """
    out = []
    # set_0 and set_1 are numpy arrays of the same shape, representing
    # the meaning of set_0 and set_1 in all models of a particular size.
    for size, (set_0, set_1) in enumerate(zip(set_repr_0, set_repr_1)):
        # Append a np.array of the same shape as set_0 and set_1.
        # For each position, put a 1, if set_0 has 1 and set_1 has 0,
        # on that position. Put a 0 otherwise.
        out.append(set_0 & (1 - set_1))
    return out


def mod_func(x: np.array, y: np.array):
    """ Compute x[i] modulo y[i] for each integer in x and y.

    Args:
        x: An np.array with integers. (A non-negative integer for each
            model in the universe.)
        y: An np.array with integers. (A non-negative integer for each
            model in the universe.)

    Returns:
        An np.array with integers. (A non-negative integer for each
             model in the universe.) Each entry i is the result
            of x[i] modulo y[i].

    """
    # Make a Boolean array for indexing.
    zero_indices = y <= 0
    y_with_ones = np.array(y[:], copy=True)
    # Put a 1 in each entry with a value <= 0.
    y_with_ones[zero_indices] = 1
    out = x % y_with_ones
    # Put a 0 in each entry that originally had a value <= 0.
    out[zero_indices] = 0
    return out


def div_func(x: np.array, y: np.array):
    """ Compute x[i] / y[i] for each integer in x and y.

    Args:
        x: An np.array with integers. (A non-negative integer for each
            model in the universe.)
        y: An np.array with integers. (A non-negative integer for each
            model in the universe.)

    Returns:
        An np.array with integers. (A non-negative integer for each
            model in the universe.) Each entry i is the result
            of x[i] / y[i].

    """
    # Make a Boolean array for indexing.
    zero_indices = y <= 0
    y_with_ones = np.array(y[:], copy=True)
    # Put a 1 in each entry with a value <= 0.
    y_with_ones[zero_indices] = 1
    out = x / y_with_ones
    # Put a 0 in each entry that originally had a value <= 0.
    out[zero_indices] = 0
    return vectorized_round(out, 2)


def init_operators(max_model_size: int, number_of_subsets=3) -> dict:
    """ Return a dictionary with all operators.

    Args:
        max_model_size: The number of objects in a model. Ranges
            from 1 to max_model_size.
        number_of_subsets: The number of subareas in the model.
            At most 4.
            3 refers to: AandB, AnotB, BnotA.
            4 refers to: AandB, AnotB, BnotA, neither.

    Returns:
        A dictionary with (operator_name, operator) pairs.

    """
    return {
        "index": Operator(
            "index",
            lambda i, s: index_func(i, s),
            (int, set),
            set,
        ),
        "diff": Operator(
            "diff",
            diff_func,  # (s1.get_set(model) - s2.get_set(model))
            (set, set),
            set,
        ),
        "subset": Operator("subset", subset_func, (set, set), bool),
        "identity": Operator("identity", identity_func, (set, set), bool),
        "non_identity": Operator("non_identity", non_identity_func, (set, set), bool),
        ">f": Operator(">f", lambda x, y: x > y, (float, float), bool),
        "=f": Operator(
            "=f", lambda x, y: vectorized_isclose(x, y), (float, float), bool
        ),
        ">": Operator(">", lambda x, y: x > y, (int, int), bool),
        ">=": Operator(">=", lambda x, y: x >= y, (int, int), bool),
        "=": Operator("=", lambda x, y: x == y, (int, int), bool),
        "/": Operator("/", div_func, (int, int), float),
        "-": Operator(
            "-", lambda x, y: x.astype(int) - y.astype(int), (int, int), int
        ),
        "+": Operator(
            "+", lambda x, y: x.astype(int) + y.astype(int), (int, int), int
        ),
        # set_repr_all is a list of np.arrays of shape 
        # (model_size, nr_of_models_of_that_model_size), for model size 
        # 0 to max_model_size.
        # Each np.array represents the object in a set per model.
        # The row operator gives a set representation for 1 model.
        "card": Operator(
            "card",
            lambda set_repr_all: np.array(
                list(
                    chain.from_iterable(
                        # Apply along column axis.
                        np.apply_along_axis(sum, 0, set_repr)
                        for set_repr in set_repr_all
                    )
                )
            ),
            (set,),
            int,
        ),
        # x_repr and y_repr are two set representations, for which the
        # same description holds as for set_repr_all.
        
        "entail_set": Operator("entail_set", 
            lambda x_repr, y_repr: [np.invert(x) | y for x, y in zip(x_repr, y_repr)],
            (set, set),
            set,
        ),
        "opp_entail_set": Operator("opp_entail_set",
            lambda x_repr, y_repr: [x | np.invert(y) for x, y in zip(x_repr, y_repr)],
            (set, set),
            set,
        ),
        "neg_entail_set": Operator("neg_entail_set", 
            lambda x_repr, y_repr: [x & np.invert(y) for x, y in zip(x_repr, y_repr)],
            (set, set),
            set,
        ),
        "neg_opp_entail_set": Operator("neg_opp_entail_set", 
            lambda x_repr, y_repr: [np.invert(x) & y for x, y in zip(x_repr, y_repr)],
            (set, set),
            set,
        ),
        "comp": Operator(
            "comp",
            lambda x_repr: [np.invert(x) for x in x_repr],
            (set,),
            set,
        ),
        "intersection": Operator(
            "intersection",
            lambda x_repr, y_repr: [x & y for x, y in zip(x_repr, y_repr)],
            (set, set),
            set,
        ),
        "nand_set": Operator(
            "nand_set",
            lambda x_repr, y_repr: [np.invert(x & y) for x, y in zip(x_repr, y_repr)],
            (set, set),
            set,
        ),
        "union": Operator(
            "union",
            lambda x_repr, y_repr: [x | y for x, y in zip(x_repr, y_repr)],
            (set, set),
            set,
        ),
        "nor_set": Operator(
            "nor_set",
            lambda x_repr, y_repr: [np.invert(x | y) for x, y in zip(x_repr, y_repr)],
            (set, set),
            set,
        ),
        "xor_set": Operator(
            "xor_set",
            lambda x_repr, y_repr: [((x | y) & np.invert(x & y)) for x, y in zip(x_repr, y_repr)],
            (set, set),
            set,
        ),
        "iff_set": Operator("iff_set",lambda x_repr, y_repr: [((x & y) | (np.invert(x) & np.invert(y))) for x, y in zip(x_repr, y_repr)],
            (set, set),
            set,
        ),
        "neg_iff_set": Operator("neg_iff_set", 
            lambda x_repr, y_repr: [((x | y) & (np.invert(x) | np.invert(y))) for x, y in zip(x_repr, y_repr)],
            (set, set),
            set,
        ),


        "and": Operator("and", lambda x, y: x & y, (bool, bool), bool),
        "nand": Operator("nand", lambda x, y: np.invert(x & y), (bool, bool), bool),
        "or": Operator("or", lambda x, y: x | y, (bool, bool), bool),
        "nor": Operator("nor", lambda x, y: np.invert(x | y), (bool, bool), bool),
        "xor": Operator("xor", lambda x, y:   ((x | y) & np.invert(x & y)), (bool, bool), bool),
        "neg": Operator("neg", lambda x: np.invert(x), (bool,), bool),
        "entail": Operator("entail", lambda x, y: (np.invert(x) | y), (bool, bool), bool),
        "opp_entail": Operator("opp_entail", lambda x, y: (np.invert(y) | x), (bool, bool), bool),
        "neg_entail": Operator("neg_entail", lambda x, y: np.invert(np.invert(x) | y), (bool, bool), bool),
        "neg_opp_entail": Operator("neg_opp_entail", lambda x, y: np.invert(np.invert(y) | x), (bool, bool), bool),
        "iff": Operator("iff", lambda x, y: (x & y) | (np.invert(x) & np.invert(y)), (bool, bool), bool),
        "neg_iff": Operator("neg_iff", lambda x, y: np.invert((x & y) | (np.invert(x) & np.invert(y))), (bool, bool), bool),
        "%": Operator("%", mod_func, (int, int), int),
    }
