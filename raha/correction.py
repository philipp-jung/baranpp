########################################
# Baran++: The Error Correction System
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# April 2019
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
import os
import math
import json
import pickle
import difflib
import unicodedata
import random
import concurrent
import multiprocessing
import logging
from typing import List, Tuple, Dict

import numpy
import sklearn.svm
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.linear_model

import auto_instance
import correctors
import mimir_dataset as dataset
import helpers
import pdep
########################################

root_logger = logging.getLogger()
# Check if there are no handlers attached to the root logger
if not root_logger.hasHandlers():
    # Configure logging with your debug logging settings
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    # Modify existing logging configuration to include debug logging settings
    root_logger.setLevel(logging.DEBUG)
    # Update format for existing handlers
    for handler in root_logger.handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))



########################################
class Correction:
    """
    The main class.
    """

    def __init__(self):
        """
        The constructor.
        """
        self.PRETRAINED_VALUE_BASED_MODELS_PATH = ""
        self.VALUE_ENCODINGS = ["identity", "unicode"]
        self.CLASSIFICATION_MODEL = "ABC"   # ["ABC", "DTC", "GBC", "GNB", "KNC" ,"SGDC", "SVC"]
        self.IGNORE_SIGN = "<<<IGNORE_THIS_VALUE>>>"
        self.VERBOSE = False
        self.SAVE_RESULTS = False
        self.ONLINE_PHASE = False
        self.LABELING_BUDGET = 20
        self.MIN_CORRECTION_CANDIDATE_PROBABILITY = 0.0
        self.MIN_CORRECTION_OCCURRENCE = 2
        self.MAX_VALUE_LENGTH = 50
        self.REVISION_WINDOW_SIZE = 5
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _value_encoder(value, encoding):
        """
        This method represents a value with a specified value abstraction encoding method.
        """
        if encoding == "identity":
            return json.dumps(list(value))
        if encoding == "unicode":
            return json.dumps([unicodedata.category(c) for c in value])

    @staticmethod
    def _to_model_adder(model, key, value):
        """
        This methods incrementally adds a key-value into a dictionary-implemented model.
        """
        if key not in model:
            model[key] = {}
        if value not in model[key]:
            model[key][value] = 0.0
        model[key][value] += 1.0

    def _value_based_models_updater(self, models, ud):
        """
        This method updates the value-based error corrector models with a given update dictionary.
        """
        # TODO: adding jabeja konannde bakhshahye substring
        if self.ONLINE_PHASE or (ud["new_value"] and len(ud["new_value"]) <= self.MAX_VALUE_LENGTH and
                                 ud["old_value"] and len(ud["old_value"]) <= self.MAX_VALUE_LENGTH and
                                 ud["old_value"] != ud["new_value"] and ud["old_value"].lower() != "n/a" and
                                 not ud["old_value"][0].isdigit()):
            remover_transformation = {}
            adder_transformation = {}
            replacer_transformation = {}
            s = difflib.SequenceMatcher(None, ud["old_value"], ud["new_value"])
            for tag, i1, i2, j1, j2 in s.get_opcodes():
                index_range = json.dumps([i1, i2])
                if tag == "delete":
                    remover_transformation[index_range] = ""
                if tag == "insert":
                    adder_transformation[index_range] = ud["new_value"][j1:j2]
                if tag == "replace":
                    replacer_transformation[index_range] = ud["new_value"][j1:j2]
            for encoding in self.VALUE_ENCODINGS:
                encoded_old_value = self._value_encoder(ud["old_value"], encoding)
                if remover_transformation:
                    self._to_model_adder(models[0], encoded_old_value, json.dumps(remover_transformation))
                if adder_transformation:
                    self._to_model_adder(models[1], encoded_old_value, json.dumps(adder_transformation))
                if replacer_transformation:
                    self._to_model_adder(models[2], encoded_old_value, json.dumps(replacer_transformation))
                self._to_model_adder(models[3], encoded_old_value, ud["new_value"])
        

    def _vicinity_based_models_updater(self, models, ud):
        """
        This method updates the vicinity-based error corrector models with a given update dictionary.
        """
        for j, cv in enumerate(ud["vicinity"]):
            if cv != self.IGNORE_SIGN:
                self._to_model_adder(models[j][ud["column"]], cv, ud["new_value"])

    def _domain_based_model_updater(self, model, ud):
        """
        This method updates the domain-based error corrector model with a given update dictionary.
        """
        self._to_model_adder(model, ud["column"], ud["new_value"])

    def _value_based_corrector(self, models, ed):
        """
        This method takes the value-based models and an error dictionary to generate potential value-based corrections.
        """
        results_list = []
        for m, model_name in enumerate(["remover", "adder", "replacer", "swapper"]):
            model = models[m]
            for encoding in self.VALUE_ENCODINGS:
                results_dictionary = {}
                encoded_value_string = self._value_encoder(ed["old_value"], encoding)
                if encoded_value_string in model:
                    sum_scores = sum(model[encoded_value_string].values())
                    if model_name in ["remover", "adder", "replacer"]:
                        for transformation_string in model[encoded_value_string]:
                            index_character_dictionary = {i: c for i, c in enumerate(ed["old_value"])}
                            transformation = json.loads(transformation_string)
                            for change_range_string in transformation:
                                change_range = json.loads(change_range_string)
                                if model_name in ["remover", "replacer"]:
                                    for i in range(change_range[0], change_range[1]):
                                        index_character_dictionary[i] = ""
                                if model_name in ["adder", "replacer"]:
                                    ov = "" if change_range[0] not in index_character_dictionary else \
                                        index_character_dictionary[change_range[0]]
                                    index_character_dictionary[change_range[0]] = transformation[change_range_string] + ov
                            new_value = ""
                            for i in range(len(index_character_dictionary)):
                                new_value += index_character_dictionary[i]
                            pr = model[encoded_value_string][transformation_string] / sum_scores
                            if pr >= self.MIN_CORRECTION_CANDIDATE_PROBABILITY:
                                results_dictionary[new_value] = pr
                    if model_name == "swapper":
                        for new_value in model[encoded_value_string]:
                            pr = model[encoded_value_string][new_value] / sum_scores
                            if pr >= self.MIN_CORRECTION_CANDIDATE_PROBABILITY:
                                results_dictionary[new_value] = pr
                results_list.append(results_dictionary)
        return results_list

    def _vicinity_based_corrector(self, models, ed):
        """
        This method takes the vicinity-based models and an error dictionary to generate potential vicinity-based corrections.
        """
        results_list = []
        for j, cv in enumerate(ed["vicinity"]):
            results_dictionary = {}
            if j != ed["column"] and cv in models[j][ed["column"]]:
                sum_scores = sum(models[j][ed["column"]][cv].values())
                for new_value in models[j][ed["column"]][cv]:
                    pr = models[j][ed["column"]][cv][new_value] / sum_scores
                    if pr >= self.MIN_CORRECTION_CANDIDATE_PROBABILITY:
                        results_dictionary[new_value] = pr
            results_list.append(results_dictionary)
        return results_list

    def _domain_based_corrector(self, model, ed):
        """
        This method takes a domain-based model and an error dictionary to generate potential domain-based corrections.
        """
        results_dictionary = {}
        sum_scores = sum(model[ed["column"]].values())
        for new_value in model[ed["column"]]:
            pr = model[ed["column"]][new_value] / sum_scores
            if pr >= self.MIN_CORRECTION_CANDIDATE_PROBABILITY:
                results_dictionary[new_value] = pr
        return [results_dictionary]

    def initialize_dataset(self, d):
        """
        This method initializes the dataset.
        """
        self.ONLINE_PHASE = True
        d.results_folder = os.path.join(os.path.dirname(d.path), "raha-baran-results-" + d.name)
        if self.SAVE_RESULTS and not os.path.exists(d.results_folder):
            os.mkdir(d.results_folder)
        d.column_errors = {}
        for cell in d.detected_cells:
            self._to_model_adder(d.column_errors, cell[1], cell)
        d.labeled_tuples = {} if not hasattr(d, "labeled_tuples") else d.labeled_tuples
        d.labeled_cells = {} if not hasattr(d, "labeled_cells") else d.labeled_cells
        d.corrected_cells = {} if not hasattr(d, "corrected_cells") else d.corrected_cells

        return d

    def initialize_models(self, d):
        """
        This method initializes the error corrector models.
        """
        # Correction store for feature creation
        corrections_features = []  # don't need further processing being used in ensembling.

        for feature in ['auto_instance', 'fd', 'llm_correction', 'llm_master']:
            corrections_features.append(feature)

        d.corrections = helpers.Corrections(corrections_features)
        d.lhs_values_frequencies = {}
        d.imputer_models = {}

        if self.VERBOSE:
            print("The error corrector models are initialized.")

    def sample_tuple(self, d):
        """
        This method samples a tuple.
        """
        remaining_column_erroneous_cells = {}
        remaining_column_erroneous_values = {}
        for j in d.column_errors:
            for cell in d.column_errors[j]:
                if cell not in d.corrected_cells:
                    self._to_model_adder(remaining_column_erroneous_cells, j, cell)
                    self._to_model_adder(remaining_column_erroneous_values, j, d.dataframe.iloc[cell])
        tuple_score = numpy.ones(d.dataframe.shape[0])
        tuple_score[list(d.labeled_tuples.keys())] = 0.0
        for j in remaining_column_erroneous_cells:
            for cell in remaining_column_erroneous_cells[j]:
                value = d.dataframe.iloc[cell]
                column_score = math.exp(len(remaining_column_erroneous_cells[j]) / len(d.column_errors[j]))
                cell_score = math.exp(remaining_column_erroneous_values[j][value] / len(remaining_column_erroneous_cells[j]))
                tuple_score[cell[0]] *= column_score * cell_score
        d.sampled_tuple = numpy.random.choice(numpy.argwhere(tuple_score == numpy.amax(tuple_score)).flatten())
        if self.VERBOSE:
            print("Tuple {} is sampled.".format(d.sampled_tuple))

    def label_with_ground_truth(self, d):
        """
        This method labels a tuple with ground truth.
        """
        d.labeled_tuples[d.sampled_tuple] = 1
        for j in range(d.dataframe.shape[1]):
            cell = (d.sampled_tuple, j)
            error_label = 0
            if d.dataframe.iloc[cell] != d.clean_dataframe.iloc[cell]:
                error_label = 1
            d.labeled_cells[cell] = [error_label, d.clean_dataframe.iloc[cell]]
        if self.VERBOSE:
            print("Tuple {} is labeled.".format(d.sampled_tuple))

    def prepare_augmented_models(self, d, datawig_use_cache, synchronous=False):
        """
        Prepare Mimir's augmented models:
        1) Calculate gpdeps and append them to d.
        2) Train auto_instance model for each column.
        """
        shape = d.dataframe.shape
        error_positions = helpers.ErrorPositions(d.detected_cells, shape, d.labeled_cells)
        row_errors = error_positions.updated_row_errors()

        self.logger.debug('Start FD profiling.')
        # calculate FDs
        inputted_rows = list(d.labeled_tuples.keys())
        df_user_input = d.clean_dataframe.iloc[inputted_rows, :]  # careful, this is ground truth.
        df_clean_iterative = pdep.cleanest_version(d.dataframe, df_user_input)
        d.fds = pdep.mine_fds(df_clean_iterative, d.clean_dataframe)
        self.logger.debug('Profiled FDs.')

        # calculate gpdeps
        shape = d.dataframe.shape
        error_positions = helpers.ErrorPositions(d.detected_cells, shape, d.labeled_cells)
        row_errors = error_positions.updated_row_errors()
        self.logger.debug('Calculated error positions.')

        d.fd_counts_dict, lhs_values_frequencies = pdep.fast_fd_counts(d.dataframe, row_errors, d.fds)
        self.logger.debug('Mined FD counts.')

        gpdeps = pdep.fd_calc_gpdeps(d.fd_counts_dict, lhs_values_frequencies, shape, row_errors, synchronous)
        self.logger.debug('Calculated gpdeps.')

        d.fd_inverted_gpdeps = {}
        for lhs in gpdeps:
            for rhs in gpdeps[lhs]:
                if rhs not in d.fd_inverted_gpdeps:
                    d.fd_inverted_gpdeps[rhs] = {}
                d.fd_inverted_gpdeps[rhs][lhs] = gpdeps[lhs][rhs]

        # normalize gpdeps per rhs
        for rhs in d.fd_inverted_gpdeps:
            norm_sum = 0
            for lhs, pdep_tuple in d.fd_inverted_gpdeps[rhs].items():
                if pdep_tuple is not None:
                    norm_sum += pdep_tuple.gpdep
            if norm_sum > 0:
                for lhs, pdep_tuple in d.fd_inverted_gpdeps[rhs].items():
                    if pdep_tuple is not None:
                        d.fd_inverted_gpdeps[rhs][lhs] = pdep.PdepTuple(pdep_tuple.pdep,
                                                                        pdep_tuple.gpdep,
                                                                        pdep_tuple.epdep,
                                                                        pdep_tuple.gpdep / norm_sum)

        if len(d.labeled_tuples) == self.LABELING_BUDGET:
            self.logger.debug('Start training DataWig Models.')
            # simulate user input by reading labeled data from the typed dataframe
            inputted_rows = list(d.labeled_tuples.keys())
            typed_user_input = d.typed_clean_dataframe.iloc[inputted_rows, :]
            df_clean_subset = auto_instance.get_clean_table(d.typed_dataframe, d.detected_cells, typed_user_input)
            for i_col, col in enumerate(df_clean_subset.columns):
                imp = auto_instance.train_cleaning_model(df_clean_subset,
                                                   d.name,
                                                   label=i_col,
                                                   time_limit=30,
                                                   use_cache=datawig_use_cache)
                if imp is not None:
                    self.logger.debug(f'Trained DataWig model for column {col} ({i_col}).')
                    d.imputer_models[i_col] = imp.predict_proba(d.typed_dataframe)
                    self.logger.debug(f'Used DataWig model to infer values for column {col} ({i_col}).')
                else:
                    d.imputer_models[i_col] = None
                    self.logger.debug(f'Failed to train a DataWig model for column {col} ({i_col}).')


    def generate_features(self, d, synchronous):
        """
        This method generates a feature vector for each pair of a data error and a potential correction.
        """
        n_workers = min(multiprocessing.cpu_count() - 1, 24)

        self.logger.debug('Start user feature generation of Mimir Correctors.')

        # Phodi
        fd_pdep_args = []
        fd_results = []

        for row, col in d.detected_cells:
            gpdeps = d.fd_inverted_gpdeps.get(col)
            if gpdeps is not None:
                local_counts_dict = {lhs_cols: d.fd_counts_dict[lhs_cols] for lhs_cols in gpdeps}  # save memory by subsetting counts_dict
                row_values = list(d.dataframe.iloc[row, :])
                fd_pdep_args.append([(row, col), local_counts_dict, gpdeps, row_values, 'norm_gpdep'])
        
        if len(fd_pdep_args) > 0:
            if synchronous:
                fd_results = map(correctors.generate_pdep_features, *zip(*fd_pdep_args))
            else:
                chunksize = len(fd_pdep_args) // min(len(fd_pdep_args), n_workers)  # makes it so that chunksize >= 0.
                with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                    fd_results = executor.map(correctors.generate_pdep_features, *zip(*fd_pdep_args), chunksize=chunksize)

        for r in fd_results:
            d.corrections.get(r['corrector'])[r['cell']] = r['correction_dict']

        self.logger.debug('Finished generating Phodi features.')
        
        # ET_CorrFM
        if len(d.labeled_tuples) == self.LABELING_BUDGET:
            error_correction_pairs: Dict[int, List[Tuple[str, str]]] = {}
            llm_correction_args = []
            llm_correction_results = []

            # Construct pairs of ('error', 'correction') per column by iterating over the user input.
            for cell in d.labeled_cells:
                if cell in d.detected_cells:
                    error = d.detected_cells[cell]
                    correction = d.labeled_cells[cell][1]
                    if error != '':
                        if correction == '':  # encode missing value
                            correction = '<MV>'
                        if error_correction_pairs.get(cell[1]) is None:
                            error_correction_pairs[cell[1]] = []
                        error_correction_pairs[cell[1]].append((error, correction))

            for (row, col) in d.detected_cells:
                old_value = d.dataframe.iloc[(row, col)]
                if old_value != '' and error_correction_pairs.get(col) is not None:  # Skip if there is no value to be transformed or no cleaning examples
                    llm_correction_args.append([(row, col), old_value, error_correction_pairs[col], d.name, d.error_fraction, d.version, d.error_class])
            
            if len(llm_correction_args) > 0:
                if synchronous:
                    llm_correction_results = map(correctors.generate_llm_correction_features, *zip(*llm_correction_args))
                else:
                    chunksize = len(llm_correction_args) // min(len(llm_correction_args), n_workers)
                    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                        llm_correction_results = executor.map(correctors.generate_llm_correction_features, *zip(*llm_correction_args), chunksize=chunksize)

            for r in llm_correction_results:
                d.corrections.get(r['corrector'])[r['cell']] = r['correction_dict']
                    
        self.logger.debug('Finished generating ET_CorrFM features.')

        # RD_ImpFM
        if len(d.labeled_tuples) == self.LABELING_BUDGET:
            llm_master_args = []
            llm_master_results = []

            error_positions = helpers.ErrorPositions(d.detected_cells, d.dataframe.shape, d.labeled_cells)
            row_errors = error_positions.updated_row_errors()
            rows_without_errors = [i for i in range(d.dataframe.shape[0]) if len(row_errors[i]) == 0]

            if len(rows_without_errors) < 3:
                llm_master_results = []
            else:
                subset = random.sample(rows_without_errors, min(100, len(rows_without_errors)))
                df_error_free_subset = d.dataframe.iloc[subset, :].copy()
                for (row, col) in d.detected_cells:
                    df_row_with_error = d.dataframe.iloc[row, :].copy()
                    llm_master_args.append([(row, col), df_error_free_subset, df_row_with_error, d.name, d.error_fraction, d.version, d.error_class])
                
                if len(llm_master_args) > 0:
                    if synchronous:
                        llm_master_results = map(correctors.generate_llm_master_features, *zip(*llm_master_args))
                    else:
                        chunksize = len(llm_master_args) // min(len(llm_master_args), n_workers)
                        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                            llm_master_results = executor.map(correctors.generate_llm_master_features, *zip(*llm_master_args), chunksize=chunksize)

            for r in llm_master_results:
                d.corrections.get(r['corrector'])[r['cell']] = r['correction_dict']

        self.logger.debug('Finished generating RD_ImpFM features.')

        # Datawig corrector
        if len(d.labeled_tuples) == self.LABELING_BUDGET:
            auto_instance_args = []
            datawig_results = []

            for (row, col) in d.detected_cells:
                df_probas = d.imputer_models.get(col)
                if df_probas is not None:
                    auto_instance_args.append([(row, col), df_probas.iloc[row], d.dataframe.iloc[row, col]])

            if len(auto_instance_args) > 0:
                if synchronous:
                    datawig_results = map(correctors.generate_datawig_features, *zip(*auto_instance_args))
                else:
                    chunksize = len(auto_instance_args) // min(len(auto_instance_args), n_workers)
                    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                        datawig_results = executor.map(correctors.generate_datawig_features, *zip(*auto_instance_args), chunksize=chunksize)

            for r in datawig_results:
                d.corrections.get(r['corrector'])[r['cell']] = r['correction_dict']

        self.logger.debug('Finished generating DataWig features.')

    def predict_corrections(self, d):
        """
        This method predicts
        """
        d.pair_features = d.corrections.assemble_pair_features()
        for j in d.column_errors:
            x_train = []
            y_train = []
            x_test = []
            test_cell_correction_list = []
            for k, cell in enumerate(d.column_errors[j]):
                if cell in d.pair_features:
                    for correction in d.pair_features[cell]:
                        if cell in d.labeled_cells and d.labeled_cells[cell][0] == 1:
                            x_train.append(d.pair_features[cell][correction])
                            y_train.append(int(correction == d.labeled_cells[cell][1]))
                            d.corrected_cells[cell] = d.labeled_cells[cell][1]
                        else:
                            x_test.append(d.pair_features[cell][correction])
                            test_cell_correction_list.append([cell, correction])
            if self.CLASSIFICATION_MODEL == "ABC":
                classification_model = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
            if self.CLASSIFICATION_MODEL == "DTC":
                classification_model = sklearn.tree.DecisionTreeClassifier(criterion="gini")
            if self.CLASSIFICATION_MODEL == "GBC":
                classification_model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100)
            if self.CLASSIFICATION_MODEL == "GNB":
                classification_model = sklearn.naive_bayes.GaussianNB()
            if self.CLASSIFICATION_MODEL == "KNC":
                classification_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
            if self.CLASSIFICATION_MODEL == "SGDC":
                classification_model = sklearn.linear_model.SGDClassifier(loss="hinge", penalty="l2")
            if self.CLASSIFICATION_MODEL == "SVC":
                classification_model = sklearn.svm.SVC(kernel="sigmoid")
            if x_train and x_test:
                if sum(y_train) == 0:
                    predicted_labels = numpy.zeros(len(x_test))
                elif sum(y_train) == len(y_train):
                    predicted_labels = numpy.ones(len(x_test))
                else:
                    classification_model.fit(x_train, y_train)
                    predicted_labels = classification_model.predict(x_test)
                # predicted_probabilities = classification_model.predict_proba(x_test)
                # correction_confidence = {}
                for index, predicted_label in enumerate(predicted_labels):
                    cell, predicted_correction = test_cell_correction_list[index]
                    # confidence = predicted_probabilities[index][1]
                    if predicted_label:
                        # if cell not in correction_confidence or confidence > correction_confidence[cell]:
                        #     correction_confidence[cell] = confidence
                        d.corrected_cells[cell] = predicted_correction
        if self.VERBOSE:
            print("{:.0f}% ({} / {}) of data errors are corrected.".format(100 * len(d.corrected_cells) / len(d.detected_cells),
                                                                           len(d.corrected_cells), len(d.detected_cells)))
    def clean_with_user_input(self, d):
        """
        User input ideally contains completely correct data. It should be leveraged for optimal cleaning
        performance.
        """
        self.logger.debug('Start cleaning with user input')
        for error_cell in d.detected_cells:
            if error_cell in d.labeled_cells:
                d.corrected_cells[error_cell] = d.labeled_cells[error_cell][1]
        self.logger.debug('Finish cleaning with user input')

    def store_results(self, d):
        """
        This method stores the results.
        """
        ec_folder_path = os.path.join(d.results_folder, "error-correction")
        if not os.path.exists(ec_folder_path):
            os.mkdir(ec_folder_path)
        pickle.dump(d, open(os.path.join(ec_folder_path, "correction.dataset"), "wb"))
        if self.VERBOSE:
            print("The results are stored in {}.".format(os.path.join(ec_folder_path, "correction.dataset")))

    def run(self, d, datawig_use_cache):
        """
        This method runs Baran++ on an input dataset to correct data errors.
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "---------------------Initialize the Dataset Object----------------------\n"
                  "------------------------------------------------------------------------")
        d = self.initialize_dataset(d)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "--------------------Initialize Error Corrector Models-------------------\n"
                  "------------------------------------------------------------------------")
        self.initialize_models(d)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "--------------Iterative Tuple Sampling, Labeling, and Learning----------\n"
                  "------------------------------------------------------------------------")
        while len(d.labeled_tuples) < self.LABELING_BUDGET:
            self.sample_tuple(d)
            self.label_with_ground_truth(d)
            self.prepare_augmented_models(d, datawig_use_cache, synchronous=False)
            self.generate_features(d, synchronous=False)
            self.predict_corrections(d)
            self.clean_with_user_input(d)

            if self.VERBOSE:
                p, r, f = d.get_data_cleaning_evaluation(d.corrected_cells)[-3:]
                print("Baran++'s performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(d.name, p, r, f))

            if self.VERBOSE:
                print("------------------------------------------------------------------------")
        if self.SAVE_RESULTS:
            if self.VERBOSE:
                print("------------------------------------------------------------------------\n"
                      "---------------------------Storing the Results--------------------------\n"
                      "------------------------------------------------------------------------")
            # self.store_results(d)
        #p, r, f = d.get_data_cleaning_evaluation(d.corrected_cells)[-3:]
        return d.corrected_cells
########################################


########################################
if __name__ == "__main__":
    dataset_name = "184"
    version = 1
    error_fraction = 5
    n_rows = 1000
    error_class = 'imputer_simple_mcar'
    data = dataset.Dataset(dataset_name, error_fraction, version, error_class, n_rows)
    data.detected_cells = data.get_errors_dictionary()
    app = Correction()
    app.VERBOSE = True
    datawig_use_cache = True
    corrected_cells = app.run(data, datawig_use_cache)
    p, r, f = data.get_data_cleaning_evaluation(corrected_cells)[-3:]
    print("Baran++'s performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(data.name, p, r, f))
########################################
