# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:33:34 2019
"""

import os

def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]
            
            
            
# -*- coding: utf-8 -*-

from os     import listdir
from numpy  import sqrt
from numpy  import log
from numpy  import sum
from numpy  import dot
from numpy  import loadtxt
from numpy  import zeros
from numpy  import eye
from numpy  import maximum
from pandas import DataFrame
from pandas import concat

MY_EPSILON = 1e-20

def buildLabeledDatabaseFromDirectory(directory_location):
    """
    @brief : build a pandas database to perform learning on subtrees.
    @param  directory_location : path to folder containing all patients to load in the database
    For each patient, load all descriptors with legends and all probabilistic labels with legend.
    The directory_location folder is expected to contain one folder per patient.
    Each patient folder is expected to contain one folder per acquisition (case).
    Each case is expected to contain the following:
    - subtree_descriptors.txt : containing descriptors for every subtree
    - subtree_label_probabilities.txt.txt : containing all labels probability for every subtree
    - computed_descriptors_meaning.txt : containing all descriptors meaning (legend)
    - label_names.txt : containing all labels meaning (legend)
    """
    differentPatients = listdir(directory_location)
    PandasDB = DataFrame()
    all_features = set({})
    all_labels = set({})
    for patDir in differentPatients:
        subtreeDir = listdir(directory_location + "/" + patDir)
        for caseDir in subtreeDir:
            currLoc2read = directory_location + "/" + patDir + "/" + caseDir
            curr_case_X = loadtxt(currLoc2read + "/subtree_descriptors.txt")
            curr_case_Y = loadtxt(currLoc2read + "/subtree_label_probabilities.txt")
            with open(currLoc2read + "/computed_descriptors_meaning.txt", "r") as descriptorsLegendFile:
                currDescriptorsLegend = descriptorsLegendFile.read().split()
                for leg in currDescriptorsLegend:
                    all_features.add(leg)
                curr_dataframe_X = DataFrame(curr_case_X, columns=currDescriptorsLegend)
                curr_dataframe_X["patient"] = patDir
                curr_dataframe_X["case"] = caseDir
            (curr_case_label_indexes, curr_case_label_names) = loadLabelLegend(currLoc2read + "/label_names.txt")
            # add read labels to the set of all labels
            for lab in curr_case_label_names:
                all_labels.add(lab)
            curr_dataframe_Y = DataFrame(curr_case_Y, columns=curr_case_label_names)
            curr_dataframe = concat([curr_dataframe_X,curr_dataframe_Y], axis=1)
            PandasDB = PandasDB.append(curr_dataframe)
    all_features = list(all_features)
    all_labels = list(all_labels)
    return (PandasDB, all_features, all_labels)

def loadLabelLegend(label_legend_location):
    res_label_legend = []
    res_label_index = []
    with open(label_legend_location, "r") as labelLegendFile:
        for line in labelLegendFile.read().splitlines():
            splitRes = line.split(" : ")
            res_label_index.append(int(splitRes[0]))
            res_label_legend.append(splitRes[1].replace(" ", "_"))
    return (res_label_index, res_label_legend)

def loadSubtreeDescriptors(subtree_descriptors_location, subtree_descriptors_meaning_location):
    descriptors = loadtxt(subtree_descriptors_location)
    with open(subtree_descriptors_meaning_location, "r") as descriptorsLegendFile:
        currDescriptorsLegend = descriptorsLegendFile.read().split()
        descriptorsDataframe = DataFrame(descriptors, columns=currDescriptorsLegend)
    return descriptorsDataframe

def computePatientLengthConfusionMatrix(patient_prediction_location, patient_ground_truth_location, labels_names_file):
    """
    @brief: Compute the patient length confusion matrix given the location of its prediction and ground truth.
    @param patient_prediction_location   : folder containing the prediction data
    @param patient_ground_truth_location : folder containing the ground truth data
    @param labels_names_file             : file containing the name of the labels (stored as integer)
    We define the length confusion matrix as the confusion matrix were branches contribute with respect to their length.
    Length is computed based on the branches stored in patient_ground_truth_location.
    The matrix is defined with the following convention:
    - each line correspond to a given prediction class
    - each column correspond to a given ground truth class
    Both folders are assumed to have a particular hierarchy:
    - The folder patient_ground_truth_location:
        * all branches named "branch????.txt"
        * a "branch_labels.txt" file
    -The folder patient_prediction_location:
        * all branches named "branch????.txt"
        * a file "recomputed_labels.txt"
    N.B. It is assumed that the number of branches in both folder are identical and that the files storing labels have the same number lines.
    """
    # Loading:
    ground_truth_couple_branchID_labelNb = loadtxt(patient_ground_truth_location + "/branch_labels.txt")
    prediction_couple_branchID_labelNb = loadtxt(patient_prediction_location + "/recomputed_labels.txt")
    (label_index, label_legend) = loadLabelLegend(labels_names_file)

    # Assert that all sizes are correct
    assert (len(prediction_couple_branchID_labelNb) == len(ground_truth_couple_branchID_labelNb))

    # Compute length for all branches
    branch_length = []
    for branch_index in prediction_couple_branchID_labelNb[:,0]:
        curr_branch = loadtxt(patient_ground_truth_location + "/branch" + format(int(branch_index), '04d') + ".txt")
        if len(curr_branch) == 0:
            # print("Ignoring empty branch in computePatientLengthConfusionMatrix")
            curr_length = 0.0
        else:
            curr_XYZ = curr_branch[:,0:-1]  # Ignore the radius when computing the length
            diff_XYZ = curr_XYZ[1:] - curr_XYZ[0:-1]
            elementary_distances = sqrt(sum(diff_XYZ * diff_XYZ, axis=1))
            curr_length = sum(elementary_distances)
        branch_length.append(curr_length)

    # Add the unknown label if not present:
    if label_index.count(-1) == 0:
        label_index.append(-1)
        label_legend.append("Unknown")

    # compute confusion matrix
    nb_labels = len(label_legend)
    nb_branches = ground_truth_couple_branchID_labelNb.shape[0]
    resulting_confusion_matrix = zeros((nb_labels, nb_labels))
    for branchID in range(nb_branches):
        int_label_GT = int(ground_truth_couple_branchID_labelNb[branchID,1])
        int_label_pred = int(prediction_couple_branchID_labelNb[branchID,1])
        index_GT = label_index.index(int_label_GT)
        index_pred = label_index.index(int_label_pred)
        resulting_confusion_matrix[index_pred, index_GT] += branch_length[branchID]

    # return the confusion matrix with legend
    return (resulting_confusion_matrix, label_legend)

def computePatientConfusionMatrix(patient_prediction_location, patient_ground_truth_location, labels_names_file):
    """
    @brief: Compute the patient confusion matrix given the location of its prediction and ground truth.
    @param patient_prediction_location   : folder containing the prediction data
    @param patient_ground_truth_location : folder containing the ground truth data
    @param labels_names_file             : file containing the name of the labels (stored as integer)
    We define the confusion matrix as the length confusion matrix with column normalization.
    It represents the repartition (ratio) of predicted labels for a given GT label.
    As for the length confusion matrix, it is defined with the following convention:
    - each line correspond to a given prediction class
    - each column correspond to a given ground truth class
    Both folders are assumed to have a particular hierarchy:
    - The folder patient_ground_truth_location:
        * all branches named "branch????.txt"
        * a "branch_labels.txt" file
    -The folder patient_prediction_location:
        * all branches named "branch????.txt"
        * a file "recomputed_labels.txt"
    N.B. It is assumed that the number of branches in both folder are identical and that the files storing labels have the same number lines.
    """
    # compute the patient length confusion matrix:
    (resulting_confusion_matrix, label_legend) = computePatientLengthConfusionMatrix(patient_prediction_location, patient_ground_truth_location, labels_names_file)

    # normalize each column:
    totalColumnLength = sum(resulting_confusion_matrix, axis=0)
    totalColumnLength = maximum(totalColumnLength, MY_EPSILON)  # prevent 0-division
    resulting_confusion_matrix /= totalColumnLength

    # return the confusion matrix with legend
    return (resulting_confusion_matrix, label_legend)

def PatientClassificationMetric(confusion_matrix):
    """
    @brief : compute the classification metric on a patient from its confusion matrix
    The metric is defined as the ratio between the length of *badly* annotated branches over the total length of branches.
    @param  confusion_matrix : length confusion matrix (as computed in utils.computePatientLengthConfusionMatrix
    """
    (nb_line, nb_col) = confusion_matrix.shape
    assert nb_line == nb_col
    total_length = sum(confusion_matrix)
    confusion_without_diagonal = confusion_matrix * (1 - eye(nb_line))
    bad_annotation_length = sum(confusion_without_diagonal)
    return (bad_annotation_length / total_length)

def plotConfusionMatrix(confusion_matrix, label_legend):
    """ Plot a length confusion matrix with legend """
    from matplotlib.pyplot import imshow, xticks, yticks, show, figure, xlabel, ylabel, colorbar
    figure()
    imshow(confusion_matrix)
    xticks(range(len(label_legend)), label_legend, rotation='vertical')
    xlabel("GT label")
    ylabel("Predicted label")
    yticks(range(len(label_legend)), label_legend)
    colorbar()
    show()

def mergeLabelsInDatabase(PandasDB, first_label_to_merge, second_label_to_merge, merged_label_name):
    assert isinstance(PandasDB, DataFrame)
    merged_column = DataFrame(PandasDB[first_label_to_merge] + PandasDB[second_label_to_merge], columns=[merged_label_name])
    PandasDB = PandasDB.drop(columns=[first_label_to_merge, second_label_to_merge])
    PandasDB = concat([PandasDB, merged_column], axis=1)
    return PandasDB

def manualFeatureGaussianization(PandasDB):
    """
    Heuristically defined transformation of features used to "Gaussianize" their distributions.
    In practice, sqrt, log and identity are used.
    """
    assert isinstance(PandasDB, DataFrame)
    identity = lambda x : x
    logCompatibleWithZero = lambda x : log(1.0 + x)
    transformation2apply = {}
    transformation2apply["length"] = sqrt
    transformation2apply["end_points_length"] = sqrt
    transformation2apply["average_radius"] = identity
    transformation2apply["stdev_radius"] = identity
    transformation2apply["average_curvature"] = identity
    transformation2apply["stdev_curvature"] = identity
    transformation2apply["invariant_moment_1"] = logCompatibleWithZero
    transformation2apply["invariant_moment_2"] = logCompatibleWithZero
    transformation2apply["invariant_moment_3"] = logCompatibleWithZero
    transformation2apply["lambda_1"] = identity
    transformation2apply["lambda_2"] = identity
    transformation2apply["lambda_3"] = identity
    transformation2apply["principal_direction_x"] = identity
    transformation2apply["principal_direction_y"] = identity
    transformation2apply["principal_direction_z"] = identity
    transformation2apply["average_endpoint_direction_x"] = identity
    transformation2apply["average_endpoint_direction_y"] = identity
    transformation2apply["average_endpoint_direction_z"] = identity
    for feature in PandasDB.columns :
        if feature in transformation2apply.keys() :
            PandasDB[feature] = PandasDB[feature].apply(transformation2apply[feature])

def extractFeatures(PandasDB, invariant="rotation_translation"):
    """
    Extract a set of features used to perform classification based on their invariance to some transformations.
    Admissible invariants are:
    - rotation_translation
    - translation
    - none
    By default, rotation and translation invariant are used
    """
    assert isinstance(PandasDB, DataFrame)
    if invariant == "rotation_translation":
        features = ["length" , "end_points_length" , "average_radius" , "stdev_radius" , "average_curvature" , "stdev_curvature" , "invariant_moment_1" , "invariant_moment_2" , "invariant_moment_3" , "lambda_1" , "lambda_2" , "lambda_3" ]
    elif invariant == "translation":
        features = ["length" , "end_points_length" , "average_radius" , "stdev_radius" , "average_curvature" , "stdev_curvature" , "invariant_moment_1" , "invariant_moment_2" , "invariant_moment_3" , "lambda_1" , "lambda_2" , "lambda_3" , "principal_direction_x" , "principal_direction_y" , "principal_direction_z" , "average_endpoint_direction_x" , "average_endpoint_direction_y" , "average_endpoint_direction_z"]
    else :
        features = ["length" , "end_points_length" , "average_radius" , "stdev_radius" , "average_curvature" , "stdev_curvature" , "invariant_moment_1" , "invariant_moment_2" , "invariant_moment_3" , "lambda_1" , "lambda_2" , "lambda_3" , "principal_direction_x" , "principal_direction_y" , "principal_direction_z" , "average_endpoint_direction_x" , "average_endpoint_direction_y" , "average_endpoint_direction_z", 'baricenter_x', 'baricenter_y', 'baricenter_z', 'weighted_baricenter_x', 'weighted_baricenter_y', 'weighted_baricenter_z']
    return DataFrame(PandasDB, columns=features)

def compute_balanced_sample_weight(Y):
    """
    Compute weights to apply to each example Y to have balanced training.
    This function extends the function sklearn.utils.class_weight.compute_class_weight
    to Y that are not 1-hot encoded but with probabilities.
    """
    sum_of_proba = sum(Y, axis=0)
    total = sum(sum_of_proba)
    nb_classes = Y.shape[1]
    class_weight = total / (float(nb_classes) * sum_of_proba + MY_EPSILON)
    sample_weight = dot(Y,class_weight)
    return(sample_weight)

def sample_weighted_mean_squared_error(Y_true, Y_pred):
    weights = compute_balanced_sample_weight(Y_true)
    sumSquared = ((Y_pred - Y_true)**2).mean(axis=1)
    weightedSum = sum(sumSquared * weights, axis=0)
    res = weightedSum / sum(weights)
    return res

def keras_sample_weighted_mean_squared_error(Y_true, Y_pred):
    import keras.backend as K
    sum_of_proba = K.sum(Y_true, axis=0)
    total = K.sum(sum_of_proba)
    nb_cases = K.shape(Y_true)[0]
    nb_classes = K.shape(Y_true)[1]
    class_weight = total / ( K.cast(nb_classes, dtype='float32') * sum_of_proba + MY_EPSILON)
    weights = K.dot(Y_true, K.reshape(class_weight, (nb_classes, -1)))
    sumSquared = K.mean((Y_pred - Y_true)**2, axis=1)
    weightedSum = K.sum(K.reshape(sumSquared, (nb_cases, -1)) * weights)
    res = weightedSum / K.sum(weights)
    return res
