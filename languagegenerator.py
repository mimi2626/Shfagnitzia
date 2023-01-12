'''
This file is part of QuantifierComplexity.
'''
import os
from pathlib import Path
import itertools as it
from collections import namedtuple
from collections import defaultdict
import time
import datetime
import json
import dotenv
import numpy as np
import pandas as pd
import dill
from pre_filter import pre_filter
import quantifier_properties
import operators
import utils


# Load environment variables from .env (which is in same dir as src).
# Don't forget to set "PROJECT_DIR" in .env to the name of the location 
# from which you are running current source code.
dotenv.load_dotenv(dotenv.find_dotenv())
# Set paths to relevant directories.
PROJECT_DIR = Path(os.getenv("PROJECT_DIR"))
LANGUAGE_SETUP_DIR_RELATIVE = os.getenv("LANGUAGE_SETUP_DIR_RELATIVE")
RESULTS_DIR_RELATIVE = os.getenv("RESULTS_DIR_RELATIVE")
LANGUAGE_SETUP_DIR = PROJECT_DIR / LANGUAGE_SETUP_DIR_RELATIVE
RESULTS_DIR = PROJECT_DIR / RESULTS_DIR_RELATIVE

Atom = namedtuple("Atom", "name func is_constant")
Atom.__repr__ = lambda self: str(self.name)

class LanguageGenerator:
    '''Class to generate quantifier language according to a grammar.'''

    def __init__(
        self, max_model_size, dest_dir=RESULTS_DIR, language_name="",
        store_at_each_length=1,
    ):
        '''Initializes LanguageGenerator object.

        Args:
            max_model_size: The size of models on which expressions will be
                evaluated.
            dest_dir: Directory in which to store results
            language_name: Name of language (corresponds to a .json
                file).
            store_at_each_length: If 1, stores expressions and itself at
                every length.

        '''
        self.date_obj = datetime.datetime.now()
        self.date = self.date_obj.strftime("%Y-%m-%d")
        self.max_model_size = max_model_size
        self.results_dir = dest_dir
        self.store_at_each_length = store_at_each_length
        self.language_name = language_name
        self.language_setup = f"{self.language_name}.json"
        self._load_json(self.language_setup)
        self._set_subset_description()
        self.N_OF_MODELS = (
            sum(
                self.number_of_subsets ** i 
                for i in range(max_model_size + 1)
            ) - 1
        )
        self._set_output_dirs()
        self._set_operators()
        self.meaning_matrix = None
        self.quantifier_prop2exp2score = dict()
        self.big_data_table = None
        self.max_expr_len = None

    def _load_json(self, language_setup):
        '''Loads .json file with language settings

        Args:
            language_setup: .json file with language settings

        '''
        if language_setup:
            with open(
                LANGUAGE_SETUP_DIR
                / f"{language_setup}",
                "r",
            ) as f:
                self.json_data = json.load(f)
                self.number_of_subsets = self.json_data["number_of_subsets"]
        else:
            self.json_data = None

    def _set_output_dirs(self):
        '''Set output directories.'''
        self.language_dir = Path(self.results_dir
        ) / utils.make_language_dir_name(
            self.max_model_size, self.language_name
        )
        self.date_dir = self.language_dir / self.date
        self.expressions_dir = self.date_dir / "expressions"
        self.language_generators_dir = self.date_dir / "language_generators"
        self.csv_dir = self.date_dir / "csv"
        self.stdout_dir = self.date_dir / "stdout"
        self.analysis_dir = self.date_dir / "analysis"
        for directory in [
            self.language_dir, self.date_dir, self.expressions_dir,
            self.language_generators_dir, self.csv_dir, self.stdout_dir, 
            self.analysis_dir
        ]:
            if not os.path.exists(directory):
                os.mkdir(directory) 

    def _set_subset_description(self):
        '''Set subset_description as an attribute. 
        
        self.subset_description:
            A string with description of subsets used in this lang gen.
            
        Note: works for the obvious choice of subsets for 
        number_of_subsets is 3 or 4. Where 4 subsets refer to all 4 
        subareas of a quantifier model. A model is defined by an 
        ordered domain M, set A \subset M, and set B \subset M. This 
        gives 4 "subsets" which refer to the areas AnotB, AandB, 
        BnotA, and neither. Number_of_subsets = 4 refers to all 
        areas, number_of_subsets = 3 refers to the areas AnotB, 
        AandB, and BnotA. For number_of_subsets is 1 or 2,
        check whether it picks the subsets that you are actually using.

        Note: computing the meaning of quantifiers is currently
        only quarenteed to work for number_of_subsets = 3. For 
        number_of_subsets = 1, 2, or 4, to work, some adjustments might 
        need to be made in generate_universe().

        '''
        all_subsets = ["AnotB", "AandB", "BnotA", "neither"]
        current_subsets_descr = str(self.number_of_subsets)
        for idx, subset in enumerate(all_subsets):
            if idx < self.number_of_subsets:
                current_subsets_descr += f"_{subset}"
        self.subset_description = current_subsets_descr
        return current_subsets_descr

    def _set_operators(self):
        '''Loads all operators in a dict.'''
        all_operators = operators.init_operators(
            self.max_model_size, self.number_of_subsets
        )
        if self.json_data is not None:
            self.operators = dict(
                (op, all_operators[op]) for op in self.json_data["operators"]
            )
        else:
            self.operators = all_operators

    def get_expr_type(self, arg):
        '''Get output type of expr, i.e. to what it would evaluate.'''
        if type(arg) is tuple:
            return self.operators[arg[0]].output_type
        if type(arg) is float:
            arg = round(arg, 2) 
        return self.atom2expr_type[str(arg)]

    def same_meaning(self, el0, el1):
        '''Determines if 2 elements have same meaning.'''
        if type(el0) != type(el1):
            out = False
        elif isinstance(el0, list):
            out = all(
                np.array_equal(set_repr0, set_repr1)
                for set_repr0, set_repr1 in zip(el0, el1)
            )
        elif isinstance(el0, np.ndarray):  # must be bitarray
            out = np.array_equal(el0, el1)
        else:
            pass
        return out

    def gen_all_expr_and_their_scores(
        self, max_expr_len: int, verbose=False, compute_scores=True
    ) -> list:
        '''Generate all semanitically unique exprs up to max_expr_len.

        The length of an expression is defined by the number of 
        operators that it contains. 

        Build expressions syntactically,
        iteratively from small to large. Start with atoms (length 0), 
        use those to build expressions with one operator (length 1),
        and use those to build expressions with two or three operators
        (length 2, length 3), etc. For ech new expression, compute
        its extension (it"s semantic meaning). The extension
        is an array that provides the meaning for the expression for 
        each model (in a given enumeration of models up to 
        max_model_size). The type of extension depends on the type of 
        expression. Set expression -> set representation 
        (collection of ordered objects); Integer expression --> integer; 
        Boolean expression -> Boolean. That extension is compared to 
        the extension of expressions of the same length that have 
        already been stored. If it is identical to any of those 
        extensions, than the expression is not semantically unique, 
        and it is not stored, if its meaning is unique, 
        the expression is stored.

        Args:
            max_expr_len: Maximum expression length. Expressions of 
                length 0 to length max_expr_len are generated.
            verbose: Print all expressions (added, filtered, not unique) 
                when True.

        Returns: 
            List with all semantically unique boolean sentences.

        '''
        # To keep track of expression that were not added, and why.
        self.filtered_expressions = {
            int: [],
            bool: [],
            set: [],
        }
        self.non_unique_expressions = {
            int: [],
            bool: [],
            set: [],
        }
        self.non_unique_expressions2originals = dict()
        self._init_atoms()
        # Keep two administrations:
        # one with all expressions and their extensions, per type
        # (self.type2expr2meaning, initialized in init_atoms);
        # another one with with all expressions per expr_len
        # (self.len2type2expr, initialized here with the atoms).
        # This last one is handy for generating all expressions,
        # iteratively, starting from atoms, and then building longer
        # expressions by combining shorter expressions already build.

        # Atoms count as expressions of length 0.
        self.len2type2expr = defaultdict(lambda: defaultdict(list))
        for expr_type, atoms in self.type2atoms.items():
            self.len2type2expr[0][expr_type] = list(
                map(lambda atom: atom.name, atoms)
            )

        # Generate expression of length 1 and larger.
        for cur_expr_len in range(1, max_expr_len + 1):
            print()
            print(f"Generating new expressions of length {cur_expr_len},",
                datetime.datetime.now()
            )
            start = time.time()
            # For each operator, build each possible expression, 
            # that has a unique meaning, i.e. for which meaning there 
            # has not yet been an expression added with that same 
            # meaning.
            for operator_name, operator in self.operators.items():
                expr_type = operator.output_type
                input_types = operator.input_types # of length 1 or 2
                # Generate expressions with a unary main operator.
                if len(input_types) == 1:
                    # Expr consists of operator and 1 argument: the 
                    # sub_expr, which is of one length shorter than 
                    # the length of expr.
                    sub_expr_type = input_types[0]
                    for sub_expr in self.len2type2expr[cur_expr_len - 1][
                        sub_expr_type
                    ]:
                        expr = (operator_name, sub_expr)
                        unique_meaning = self._get_meaning_if_unique(expr)
                        if unique_meaning is not None:
                            # Add expr (operator_name, sub_expr) and 
                            # its meaning.
                            self.len2type2expr[cur_expr_len][expr_type].append(
                                expr
                            )
                            self.type2expr2meaning[expr_type][expr] = \
                                unique_meaning

                # Generate expressions with a binary main operator,
                # by combining 2 subexpressions (arguments).
                elif len(input_types) == 2:
                    # The length of both subexpressions combined must be 
                    # cur_expr_len - 1.
                    for arg_len_1 in range(0, cur_expr_len):
                        arg_len_2 = cur_expr_len - arg_len_1 - 1
                        arg_type_1 = input_types[0]
                        arg_type_2 = input_types[1]
                        for sub_expr_1 in self.len2type2expr[
                            arg_len_1
                        ][arg_type_1]:
                            for sub_expr_2 in self.len2type2expr[
                                arg_len_2
                            ][arg_type_2]:
                                expr = (operator_name, sub_expr_1, sub_expr_2)
                                unique_meaning = self._get_meaning_if_unique(
                                    expr
                                )
                                if unique_meaning is not None:
                                    # Add expr (operator_name, sub_expr) 
                                    # and its meaning.
                                    self.len2type2expr[
                                        cur_expr_len
                                    ][expr_type].append(expr)
                                    self.type2expr2meaning[expr_type][
                                        expr
                                    ] = unique_meaning
            self.max_expr_len = cur_expr_len
            new = sum(map(len, self.len2type2expr[cur_expr_len].values()))
            if compute_scores:
                self.get_big_data_table()
            if self.store_at_each_length:
                print()
                self.write_expressions()
                # Compute and store all scores for all expressions so 
                # far.
            finish = time.time()
            print("running_time =", (finish-start) / 60)

        def gen_lang_admin(expr_type, verbose=False, verbose_non_unique=False):
            print(
                str(len(self.type2expr2meaning[expr_type])), 
                str(expr_type)[8:-2] +" expressions added:"
            )
            if verbose:
                for expression in self.type2expr2meaning[expr_type]:
                    print(utils.pretty_print_expr(expression))
                print()
            print(
                str(len(self.filtered_expressions[expr_type])), 
                str(expr_type)[8:-2] +" expressions filtered:"
            )
            if verbose:
                for expression in self.filtered_expressions[expr_type]:
                    print(utils.pretty_print_expr(expression))
                print()
            print(
                str(len(self.non_unique_expressions[expr_type])), 
                str(expr_type)[8:-2] +" expressions not unique:"
            )
            if verbose or verbose_non_unique:
                for expression in self.non_unique_expressions[expr_type]:
                    print(
                        utils.pretty_print_expr(expression), "\t-->\t ",
                        utils.pretty_print_expr(
                            self.non_unique_expressions2originals[expression]
                        )
                    )
                print()
            print()

        def gen_lang_admin_cummulative(expr_dict, action: str):
            print(
                sum(map(
                    len, [expr_dict[int], expr_dict[set], expr_dict[bool]])
                ), "expressions " + action
            )

        if verbose:
            print()
            gen_lang_admin(int, verbose=False, verbose_non_unique=True)
            gen_lang_admin(set, verbose=False, verbose_non_unique=True)
            gen_lang_admin(bool, verbose=False, verbose_non_unique=True)
        print("\n*SUMMARY*\n")  
        print("MAX_EXPR_LEN =", max_expr_len)
        print("MAX_MODEL_SIZE =", self.max_model_size, "\n")
        gen_lang_admin(int)
        gen_lang_admin(set)
        gen_lang_admin(bool, )
        gen_lang_admin_cummulative(self.type2expr2meaning, "added")
        gen_lang_admin_cummulative(self.filtered_expressions, "filtered")
        gen_lang_admin_cummulative(self.non_unique_expressions, "not unique")
        print()
        return list(self.type2expr2meaning[bool])

    def _init_atoms(self):
        '''Generate all atoms and their meaning.

        Initializes self.type2expr2meaning and stores the atoms (Set 
        variables A and B, and integer constants 0 to max_model size) 
        and their meaning there. Method gen_all_expr will store all 
        other expressions and their meanings there.

        An atom ia namedtuple. Atom.name should be a string.
        Set variables are model-dependent. Integer constants are
        model-invariant.

        '''
        self.type2expr2meaning = dict(
            (expr_type, dict()) for expr_type in [bool, int, set]
        )
        type2atoms = {
            int: [],
            set: [],
        }
        # Initialize the integer atoms.
        for integer in range(0, self.max_model_size + 1, 1):
            type2atoms[int].append(
                Atom(str(integer), lambda i=integer: i, True)
            )
        # Initialize the set atoms.
        set_name2relevant_index = {"A": 0, "B": 1}
        for set_name in ["A", "B"]:
            rel_idx = set_name2relevant_index[set_name]
            type2atoms[set].append(
                Atom(
                    set_name,
                    lambda model, i=rel_idx: np.array(
                        [int(obj) for obj in model[i]]
                    ),
                    False,
                )
            )
        # Create usefull acces from properties to atoms and vice versa.
        self.type2atoms = type2atoms
        self.atom2expr_type = dict()
        for expr_type, atoms in type2atoms.items():
            for atom in atoms:
                self.atom2expr_type[atom.name] = expr_type
        all_atoms = list(it.chain.from_iterable(type2atoms.values()))
        self.name2atom = dict(
            (atom.name, atom)
            for atom in all_atoms
        )
        # Compute and store meaning (extension) of atoms.
        for atom in all_atoms:
            atom_meaning = self._compute_meaning_atom(atom)
            self.type2expr2meaning[self.get_expr_type(atom)][
                atom.name
            ] = atom_meaning

    def _compute_meaning_atom(self, atom: namedtuple, debug=False):
        ''' Computes the meaning of an atom for all models.

        The meaning of integer constants are model invariant:
        its meaning is the same in each model: namely, that integer 
        itself. The meaning of set variables are model dependent: it
        provides, given an enumeration of model, the information of,  
        given a model in that enumeration, which elements of that 
        model are present in that set.

        Args:
            atom: A namedtuple created by "Atom". Either set variable
                A or B, or an integer constant from 0 to 
                max_model_size.

        Returns: 
            For integer constants: an immutable np.array of shape 
            (1,N_OF_MODELS), for set variables: a list of 
            self.max_model_size many immutable np.arrays of shape 
            (model_size, number_of_subsets**model_size) for model_size 
            from 1 to max_model_size.

        '''
        # Model-invariant atom case: integer constants.
        # An array with that same integer for each model in the 
        # universe. 
        if atom.is_constant:
            atom_meaning = np.array(
                [atom.func()] * self.N_OF_MODELS
            )
            atom_meaning.flags.writeable = False
        # Model-dependent atom case: set variables.
        # A set representation shows which elements are in a set
        # accross the whole universe of models.
        else:
            atom_meaning = []
            cur_model_size = 1
            # Make matrix template of the correct shape. 
            # Columns represent models, rows represent objects.
            # Entry (i,j) represents whether element i in model j 
            # is in A (1) or not in A (0).
            # For model_size 0 to max_model_size there will be a matrix 
            # of shape (model_size, number_of_subsets ** model_size) 
            # in set_repr.
            number_of_models_of_cur_model_size = (
                self.number_of_subsets ** cur_model_size
            )
            set_repr = np.zeros(
                (cur_model_size, number_of_models_of_cur_model_size), 
                dtype=np.uint8
            )  
            # Offset = number of models of previous model sizes.
            offset = 0
            # Fil in the matrix template, per model.
            start_of_new_model_block = (
                self.number_of_subsets ** cur_model_size
            )
            for model_idx, model in enumerate(self.generate_universe()):
                # When all models of cur_model_size have been treated,
                # go to the next block of models: the models of size
                # cur_model_size + 1. Store the results of current
                # block of models.
                if debug:
                    print()
                    print("start_of_new_model_block = ", end="")
                    print(start_of_new_model_block)
                    print("model_idx, offset, model = ", end="")
                    print(model_idx, offset, utils.tuple_format(model))
                if model_idx == start_of_new_model_block:
                    if debug:
                        print()
                        print()
                    cur_model_size += 1
                    offset = start_of_new_model_block 
                    start_of_new_model_block += (
                        self.number_of_subsets ** cur_model_size
                    )
                    set_repr.flags.writeable = False
                    atom_meaning.append(set_repr)
                    # Reset matrix template: make new matrix template 
                    # of the correct shape.
                    set_repr = np.zeros(
                        (cur_model_size, 
                            self.number_of_subsets ** cur_model_size
                        ),
                        dtype=np.uint8,
                    )
                # For object 1 to object current_model_size in the 
                # current model, fill in set-inclusion for this 
                # set-variable. Fill in collumn i = model_index - offset 
                # (the i"th model of current_model_size).
                if debug:
                    print("model =", model)
                    print("atom.func(model) =", atom, atom.func(model))
                    print("set_repr =", set_repr)
                set_repr[:, model_idx - offset] = atom.func(model)  
                if debug:
                    print("set_repr =", set_repr)
            # Finish.
            set_repr.flags.writeable = False
            atom_meaning.append(set_repr)
        return atom_meaning

    def generate_universe(self):
        '''Return generator object that yields all models.
        
        Generates all models from size 1 to size max_model_size.
        A model is defined by an ordered domain M, set A \subset M, and 
        set B \subset M. This gives 4 "subsets" which refer to the 
        area"s AnotB, AandB, BnotA, and neither. Number_of_subsets = 4 
        refers to all area"s, Number_of_subsets = 3 refers to 
        the area"s AnotB, AandB, and BnotA.

        A model of size n is represented by a tuple of 2 lists of 
        length n, where the i-the entry represents the i-th object
        in the model. The 1st list represents inclusion in set A, 
        and the 2nd list repersents inclusion in B.
        Zip over these two lists, to get an enumeration of the 
        objects in the model.

        The models are generated in lexicographical order over
        the different "subsets" (area"s), where each object is
        represented by a character that represents the subset to
        which it belongs.

        To ensure this lexicographical order and at the same time
        keep the already existing construction of representing set A 
        and set B seperately, the used algorithm is not the most direct 
        way of generating the models.

        '''
        for model_size in range(1, self.max_model_size + 1):
            for model in it.product(
                [(0, 0), (0, 1), (1, 0), (1, 1)], repeat=model_size
            ):
                A_repr = [obj[0] for obj in model]
                B_repr = [obj[1] for obj in model]
                # A_repr = bitarray.bitarray(A_repr)
                # B_repr = bitarray.bitarray(B_repr)
                if self.number_of_subsets == 4:
                    yield (A_repr, B_repr)
                elif self.number_of_subsets == 3:
                    if all(any((a, b)) for (a, b) in zip(A_repr, B_repr)):
                        yield (A_repr, B_repr)

    def _get_meaning_if_unique(self, expr: tuple, verbose=False):
        '''Return meaning (extension) of expr if semantically unique.

        First filter based on syntax. Then compute the meaning of expr
        to the meanings of expressions already added to the lang.

        Args:
            expr: A tuple. Quantifier expression generated by 
                gen_all_expr. A tuple with strings.

        Returns:
            The return value. None if expr is not unique,
            the meaning (extension) of expr otherwise.

        '''
        expr_type = self.get_expr_type(expr)
        # Filter out tautologies and falsehoods, 
        # before computing the meaning.
        # if self.language_name in ["Logical", "Logical_index"]:
        if pre_filter(self, expr):
            if verbose:
                print("prefilter", expr)
            self.filtered_expressions[expr_type].append(expr)
            return None
        this_meaning = self.compute_meaning(expr)
        if not verbose:
            # Alternative version, to find the original expr to which
            # the current one is equivalent.
            for prev_expr, prev_meaning in \
                self.type2expr2meaning[expr_type].items():
                if self.same_meaning(this_meaning, prev_meaning):
                    self.non_unique_expressions[expr_type].append(expr)
                    self.non_unique_expressions2originals[expr] = prev_expr
                    return None    
        if verbose:
            for key, value in self.type2expr2meaning[expr_type].items():
                prev_meaning_expr = key
                prev_meaning = value
                if self.same_meaning(this_meaning, prev_meaning):
                    print(
                        "** non-unique", expr, "  eq. to", 
                        prev_meaning_expr
                    )
                    self.non_unique_expressions[expr_type].append(expr)
                    return None
        # Expr has unique meaning.
        if verbose:
            print("---> include", expr)
        return this_meaning

    def test_prefilter(self):
        '''Print incorrectly filtered exprs when present.
        
        Check wether prefilter only filters out exprs for which there
        is a shorter or equal length equivalent expr (i.e. with the 
        same meaning / extension) added to the language.

        '''
        # Custom list of safely filtered expressions that are not 
        # equivalent to any expression in the language.
        custom_exprs = [
            # The empty set may safely be filtered out, because for
            # any expression with the empty set, there is a shorter
            # equivalent expr without the empty set.
            ('diff', 'A', 'A') 
        ]
        # Testing.
        for test_type in [int, set, bool]:
            print("=== Testing filter for type {} ===".format(test_type))
            # List with (expr, expr_meaning) tuples of exprs in the 
            # language.
            lang_exprs = list(self.type2expr2meaning[test_type].keys())
            for filtered_expr in self.filtered_expressions[test_type]:
                filtered_meaning = self.compute_meaning(filtered_expr)
                found_equivalent_expr = False
                found_longer_equivalent_expr = False
                longer_equivalent_expr = None
                for test_expr in custom_exprs + lang_exprs:
                    if test_expr in custom_exprs:
                        test_meaning = self.compute_meaning(test_expr)
                    else:
                        test_meaning = self.type2expr2meaning[test_type][
                            test_expr
                        ]
                    if self.same_meaning(filtered_meaning, test_meaning):
                        found_equivalent_expr = True
                        if utils.expr_len(test_expr) > utils.expr_len(
                            filtered_expr
                        ):
                            found_longer_equivalent_expr = True
                            longer_equivalent_expr = test_expr
                        break
                if not found_equivalent_expr:
                    print(
                        "WARNING: filtered {} but no equiv expr added.".format(
                        filtered_expr
                    ))
                if found_longer_equivalent_expr: 
                    print(
                        "WARNING: filtered \t{}\n"
                        "\tbut equiv expr \t{} in lang is longer.".format(
                        filtered_expr, longer_equivalent_expr
                    ))
        print("=== Done testing ===")

    def compute_meaning(self, expr: tuple, debug=False):
        '''Return meaning representation for expr.

        Get the (already stored) meaning of the args of expr.
        Compute the meaning of expr, based on these meanings
        as input to the main operator of expr.

        Args:
            A tuple. Quantifier expression generated by gen_all_expr. 
                A tuple with strings.
            debug: A Boolean. Prints testing statements when True.
        
        Returns:
            The meaning (extension) of expr. The meaning type
            depends on the expr type.
        '''
        # Testing.
        if debug:
            print("expression = ", )
            print(expr)
        # Case: expression is an atom (namedtuple). Meaning already 
        # computed in _init_atoms, by _compute_meaning_atom.
        if not isinstance(expr, tuple):
            return self.type2expr2meaning[
                self.atom2expr_type[expr]
            ][expr]
        main_operator = self.operators[expr[0]]
        args = expr[1:]
        arg_meanings = []
        for arg in args:
            expr_type = self.get_expr_type(arg)
            # Meaning (extension) of args should already be present.
            try:
                arg_meanings.append(
                    self.type2expr2meaning[expr_type][arg]
                )
            except:  
                # Should only occur when evaluating a "new "meaning, not
                # created in this algorithm. 
                print("*********EXCEPT********")
                new_meaning = self.compute_meaning(arg)
                arg_meanings.append(new_meaning)
        # Testing.
        if debug:
            print("arg_meanings =", arg_meanings)
            print("*arg_meanings =", *arg_meanings)
        # Meaning of expr is computed based on meaning args.
        return main_operator.func(*arg_meanings)

    def write_expressions(self):
        '''Writes current expressions and itself to dill file'''
        print("Interim saving of current state")
        current_max_len = max(self.len2type2expr.keys())
        current_expressions = list(
            it.chain.from_iterable(
                type2expressions[bool]
                for type2expressions in self.len2type2expr.values()
            )
        )
        with open(
            self.expressions_dir
            / ("expressions_up_to_length_%s.dill" % current_max_len),
            "wb",
        ) as f:
            dill.dump(current_expressions, f)
        with open(
            self.language_generators_dir
            / ("language_generator_up_to_length_%s.dill" % current_max_len),
            "wb",
        ) as f:
            dill.dump(self, f)

    def get_big_data_table(self):
        '''Return dataframe with all scores and language info.'''
        if self.big_data_table is None:
            self._create_big_data_table()
            return self.big_data_table
        if self.max_expr_len > int(self.big_data_table["expr_length"].max()):
            self._create_big_data_table()
        return self.big_data_table

    def _create_big_data_table(self):
        '''Compute all scores, collect in dataframe and store as csv.'''
        data = pd.concat(
            [self.get_exp2score(quantifier_properties.Monotonicity), 
                self.get_exp2score(quantifier_properties.Quantity),
                self.get_exp2score(quantifier_properties.Conservativity),
                self.get_exp2score(quantifier_properties.LempelZiv),
                self.get_exp2score(quantifier_properties.Uniformity)
            ], axis=1
        ).reset_index()
        data.columns = [
            "expression", "monotonicity", "quantity", 
            "conservativity", "lempel_ziv", "uniformity"
        ]
        # When using lambda on pd.DataFrame, axis 1, it gives as input
        # to lambda function, each row, transposed, as a series.
        data.insert(loc=0, column="expr_length", 
            value=data.apply(
                lambda row_series: utils.get_exp_len(row_series.expression), 
                axis=1
            )
        )
        # Add extra "admin properties" to identify from which language
        # generator this data stems.
        data["max_model_size"] = self.max_model_size
        data["lot"] = self.language_name
        data["subsets"] = self.subset_description
        current_date = datetime.datetime.now().strftime("%d-%m-%Y")
        data["date"] = current_date
        data.sort_values(by=["expr_length"], inplace=True)
        data.reset_index(drop=True, inplace=True)
        max_expr_len = data["expr_length"].max()
        filename = utils.make_csv_filename(
            self.max_model_size, max_expr_len, self.language_name
        )
        file_loc = utils.make_date_dir(self.csv_dir)
        data.to_csv(file_loc / filename, index=False)
        self.big_data_table = data
        return self.big_data_table

    def get_exp2score(self, quantifier_property, verbose=False):
        '''Return graded measure scores for universal quan property.

        Args:
            quantifier_property: a subclass of QuantifierProperty.

        '''
        if quantifier_property.name == "monotonicity":
            mon_sign_up = quantifier_property.signature_function_up
            mon_sign_down = quantifier_property.signature_function_down
            self._compute_exp2univ_prop(mon_sign_up, "monotonicity_up")
            self._compute_exp2univ_prop(mon_sign_down, "monotonicity_down")
            # Take max of monotonicity_up and monotonicity_down.
            mon_up_and_down = pd.concat([
                self.quantifier_prop2exp2score[
                    "monotonicity_up"],
                self.quantifier_prop2exp2score[
                    "monotonicity_down"]
            ], axis=1)
            mon_up_and_down.columns = ["up", "down"]
            if verbose:
                print("*************************** mon_up_and_down")
                print(mon_up_and_down)
                print()
            self.quantifier_prop2exp2score[
                quantifier_property.name] = mon_up_and_down.max(axis=1)
            monotonicity_direction = mon_up_and_down.idxmax(axis=1) 
            self.quantifier_prop2exp2score[
                "monotonicity_direction"
            ] = monotonicity_direction
        elif quantifier_property.name in {"quantity", "conservativity"}:
            self._compute_exp2univ_prop(
                quantifier_property.signature_function, 
                quantifier_property.name
            )
        elif quantifier_property.name in {"lempel_ziv", "uniformity"}:
            self._compute_exp2quan_prop(quantifier_property)
        return self.quantifier_prop2exp2score[quantifier_property.name]

    def _compute_exp2univ_prop(
        self, signature_function, property_name: str
    ) -> pd.Series:
        '''Compute graded measure scores for universal quan property.

        Graded measure based on the information theoretic concepts of
        entropy and mutual information.

        For quantifier Q:
        - 1_Q: model_space -> {0,1} = random variabele: whether Q is
        true in model M;
        - Gamma_X = random variable that gives signature for
        property X, given a model, or given (model, expression);
        - H(1_Q) = entropy of 1_Q;
        - H(1_Q)|Gamma_Q) = conditional entropy of H(1_Q), given
        Gamma_X.
        - Below, we call Gamma_X: Signature

        Score(Q) = 1 - (H(1_Q)|Gamma_Q) / H(1_Q))

        Args:
            signature_function: A function that, given a meaning 
                matrix, returns all signature values for a particular 
                quantifier property. One signature value for each model, 
                or for each (model, expression) pair.
            property_name: A string. Refers to quantifier property for 
                which graded measures will be computed. Is used as key 
                in dictionary, where the results are stored as the value
                for that key. Property_name is either "monotoncity",
                "quantity", or "conservativity".

        Returns: A pandas Series, with for each expression the 
            graded measure for the given quantifier property. 
            This Series is also stored in the dictionary 
            self.quantifier_prop2exp2score, under the key
            property_name.

        '''
        meaning_matrix = self.get_meaning_matrix()
        # Get signature values for this meaning matrix.
        signatures = signature_function(meaning_matrix)
        # Case: quantity or conservativity: signature value per model 
        # (expression invariant). When signatures is a vector.
        if len(signatures.shape) == 1:
            # Add signature as a column to allow for groupbys.
            meaning_matrix["signature"] = signatures.values
            # Compute for each signature, the probability of it occurring
            # p(signature_val)
            signature2prob = meaning_matrix["signature"].value_counts(
                normalize=True).sort_index()
            # Compute matrix where the element in position (i, j) 
            # denotes the probability of the jth expression being true 
            # in a model given that the model is of the ith signature 
            # type p(M \in Q | Signature = signature_val).
            signature_exp2cond_prob = (
                meaning_matrix.groupby("signature").mean().sort_index()
            )
            del meaning_matrix["signature"]
        # Case: monotonicity: signature value per (model, expression) 
        # pair. Signature value is True of False.
        else: 
            # Compute for each signature, the probability of it 
            # occurring. p(signature_val).
            signature2prob = signatures.apply(
                lambda s: s.value_counts(normalize=True),
            ).fillna(0)
            # All signatures in an array.
            all_signatures = sorted(np.unique(signatures.values.ravel()))
            # Compute for each signature value, the probability of
            # an expression being true in a model given signature.
            # p(M \in Q | Signature = signature_val)
            signature_exp2cond_prob = pd.DataFrame(
                index=meaning_matrix.columns
            )
            for signature_val in all_signatures:
                # Mask is a dataframe like signatures, but with
                # (model, expression) set to True when 
                # signature(model, expression) = signature_val,
                # and to False otherwise.
                # So Mask = signatures, when signature_val is True,
                # the flipped version of signatures otherwise.
                mask = signatures == signature_val
                # For each expression count number of models with this
                # signature. Df Series with value for each expression.
                exp2nof_signature_occurrences = mask.sum()
                # For each expression count number of models 
                # with this signature that also satisfy the expression.
                exp2nof_true_model_occurrences = (meaning_matrix & mask).sum()
                # For each expression, compute its probability of being
                # true, conditional on this signature value.
                # p(M \in Q | Signature = signature_val) =
                # the nr of model with this signature that make Q true,
                # devided by the total nr of models iwth this signature.
                exp2cond_prob = (
                    exp2nof_true_model_occurrences
                    / exp2nof_signature_occurrences
                )
                signature_exp2cond_prob[
                    signature_val
                ] = exp2cond_prob
            # Replace nans with 0.
            signature_exp2cond_prob.fillna(0, inplace=True)
            # Take transpose so the df is indexed by signature values, 
            # and columns are the expressions
            signature_exp2cond_prob = (
                signature_exp2cond_prob.T
            )
        # Now compute the binary entropy for each 
        # p(M \in Q|Signature = signature_value).
        # H(1_Q | Signature = signature_val) = 
        # bin_entropy(p(M \in Q|Signature = signature_value)).
        signature_exp2cond_entropy = utils.element_wise_binary_entropy(
            signature_exp2cond_prob
        )
        # For the conditional entropy values for each 
        # (signature,expression) pair, compute the "weighted" entropy 
        # value by multiplying with the probability of that signature 
        # H(1_Q | Signature) = \Sigma_signature_val[
        # occurring. 
        # p(signature_val) * H(1_Q | Signature = signature_val)].
        signature_exp2cond_entropy_weighted_sum = \
            signature_exp2cond_entropy.mul(signature2prob, axis="index").sum()
        # Series that maps each expression to its probability of being 
        # true.
        exp2prob = meaning_matrix.mean()
        # Map each expression to its entropy: H(1_Q).
        exp2entropy = utils.element_wise_binary_entropy(exp2prob)
        # Now we compute the "normalized" score by dividing the 
        # conditional entropy by the actual entropy. If this value is 
        # close to zero then coming to know the signature value gives 
        # a lot of information on the truth value of the expression in 
        # a given model. H(1_Q)|Signature) / H(1_Q).
        exp2normalized_cond_entropy = (
            signature_exp2cond_entropy_weighted_sum / exp2entropy
        ).fillna(0)
        # Now we subtract the scores from 1, to make sure that those
        # expressions for which scores are low (i.e. signatures tell 
        # us a lot) are mapped to values closer to 1.
        exp2score = 1 - exp2normalized_cond_entropy
        self.quantifier_prop2exp2score[property_name] = exp2score

    def get_meaning_matrix(self):
        '''Return the most up to date meaning matrix.'''
        if self.meaning_matrix is None:
            self._compute_meaning_matrix()
            return self.meaning_matrix
        mm_exprs = self.meaning_matrix.columns.values
        mm_max_expr_len = int(
            max([utils.get_exp_len(expr) for expr in mm_exprs])
        )
        if self.max_expr_len > mm_max_expr_len:
            self._compute_meaning_matrix()
        return self.meaning_matrix

    def _compute_meaning_matrix(self):
        '''Put extensions of expressions in a matrix.

        Compute matrix where the cell in the ith row and jth column is 
        1 iff the ith model belongs to the jth expression/quantifier.
        
        Returns: None, but sets self.meaning_matrix accordingly

        '''
        # Sort all expressions that evaluate to a boolean
        sorted_exps = sorted(
            list(self.type2expr2meaning[bool]),
            key=str,
            reverse=True,
        )
        print("sorted_exps")
        print(sorted_exps)
        out = np.zeros((self.N_OF_MODELS, len(sorted_exps)))
        # For each ith expression, set the ith column to the meaning  
        # vector for that expression.
        for i, exp in enumerate(sorted_exps):
            out[:, i] = self.type2expr2meaning[bool][exp]
        # Convert to df, such that it is indexed by the models, and 
        # column names are the expressions.
        out = pd.DataFrame(
            out,
            columns=sorted_exps,
            index=[utils.tuple_format(m) for m in self.generate_universe()],
        )
        self.meaning_matrix = out.astype(int)

    def _compute_exp2quan_prop(self, quan_prop):
        '''Compute score for quantifier property per quan extension. 
        
        Store (expression, quan_prop_score) in dictionary.
        Store that dictionary in dict quantifier_prop2exp2score,
        under the name of the quantifier property.

        Args:
            quan_prop: Needs to be a subclass of 
                QuantifierProperty that has a property_function.
                Either LempelZiv or Uniformity.

        '''
        meaning_matrix = self.get_meaning_matrix()
        expr2prop = pd.Series(index=meaning_matrix.columns)
        exprs = self.type2expr2meaning[bool]
        for expr, meaning in exprs.items():
            expr2prop[expr] = quan_prop.property_function(meaning)
        self.quantifier_prop2exp2score[
            quan_prop.name
        ] = expr2prop
