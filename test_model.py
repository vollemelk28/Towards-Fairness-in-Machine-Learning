import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from keras_facenet import FaceNet
from data_loader import load_and_process_data
import os
import gc
import itertools
from itertools import combinations


TEST_DIR1 = 'HERE_YOUR_TEST_DIR'
TEST_DIR2 = 'SECOND_TEST_DIR' # For experimenting

IMG_SIZE = 144
ROC_score = []
all_f1 = []
all_di = []
all_acc = []
all_eqodds = []
all_avgeqodds = []

def Average(list): 
    return sum(list) / len(list)

def calculate_disparate_impact(tp_protected, fp_protected, tn_protected, fn_protected,
                               tp_unprotected, fp_unprotected, tn_unprotected, fn_unprotected):

    protected_group_positive = tp_protected + fp_protected
    unprotected_group_positive = tp_unprotected + fp_unprotected
    protected_group_total = tp_protected + fp_protected + tn_protected + fn_protected
    unprotected_group_total = tp_unprotected + fp_unprotected + tn_unprotected + fn_unprotected

    protected_group_prop = protected_group_positive / protected_group_total
    unprotected_group_prop = unprotected_group_positive / unprotected_group_total
    return protected_group_prop / unprotected_group_prop

def eq_odds(measures_df, cm):

    white_eqodds = 0
    black_eqodds = 0
    asian_eqodds = 0
    indian_eqodds = 0
    eq_odds_diffs = []
    avg_eq_odds_dif = 0
    tpr = measures_df.loc['TP'] / (measures_df.loc['TP'] + measures_df.loc['FN'])
    fpr = measures_df.loc['FP'] / (measures_df.loc['FP'] + measures_df.loc['TN'])

    for i in range(4):
        white_eqodds = white_eqodds + max(abs(tpr[i] - tpr[0]), abs(fpr[i] - fpr[0])) 
        black_eqodds = black_eqodds + max(abs(tpr[i] - tpr[1]), abs(fpr[i] - fpr[1]))
        asian_eqodds = asian_eqodds + max(abs(tpr[i] - tpr[2]), abs(fpr[i] - fpr[2]))
        indian_eqodds = indian_eqodds + max(abs(tpr[i] - tpr[3]), abs(fpr[i] - fpr[3]))

    print("white_eqodds: ", white_eqodds)
    print("indian_eqodds: ", indian_eqodds)
    eq_odds_diffs.append(white_eqodds)
    eq_odds_diffs.append(black_eqodds)
    eq_odds_diffs.append(asian_eqodds)
    eq_odds_diffs.append(indian_eqodds)
    avg_eq_odds_dif = Average(eq_odds_diffs)

    return eq_odds_diffs, avg_eq_odds_dif

def test_mod(model, testset):

    if testset == 1:
        TEST_DIR = TEST_DIR1
    elif testset == 2:
        TEST_DIR = TEST_DIR2

    image_paths = []
    age_labels = []
    gender_labels = []
    ethnicity_labels = []

    for filename in os.listdir(TEST_DIR):
        image_path = os.path.join(TEST_DIR, filename)
        temp = filename.split('_')
        age = int(temp[0])
        gender = int(temp[1])
        ethnicity = int(temp[2])
        image_paths.append(image_path)
        age_labels.append(age)
        gender_labels.append(gender)
        ethnicity_labels.append(ethnicity)

    # convert to dataframe
    df_test = pd.DataFrame()
    df_test['image'], df_test['ethnicity'] = image_paths, ethnicity_labels

    ethnicity_dict = {0:'White', 1:'Black', 2:'Asian', 3:'Indian', 4:'Others'}

    X_test = extract_features(df_test['image'])

    #print(np.shape(X_test))

    X_test = X_test/255.0

    y_ethnicity = np.array(df_test['ethnicity'])
    y_ethnicity_onehot = to_categorical(y_ethnicity, num_classes=4)  # Assuming you have 4 classes for ethnicity

    del y_ethnicity # Free up memory
    
    gc.collect()

    # get probabilities, not class labels
    pred_probs = model.predict(x=X_test)

    # to get class labels, use np.argmax but save it in a different variable 
    pred_labels2 = np.argmax(pred_probs, axis=1)

    true_labels2 = np.argmax(y_ethnicity_onehot, axis=1)
    true_labels = true_labels2.tolist()
    pred_labels = pred_labels2.tolist()

    ##### Equal Odds - Begin

    # # Create a container for all the rates
    # rates = {}

    # # Calculate rates for each group
    # for group in range(4):  # change the range accordingly
    #     y_true_group = true_labels2[y_ethnicity == group]
    #     y_pred_group = pred_labels2[y_ethnicity == group]
    #     tpr, fpr = calculate_rates(y_true_group, y_pred_group)
    #     rates[group] = (tpr, fpr)

    # # Calculate Equal Odds Difference for each pair of groups
    # for group1, group2 in itertools.combinations(rates.keys(), 2):
    #     tpr_group1, fpr_group1 = rates[group1]
    #     tpr_group2, fpr_group2 = rates[group2]
    #     equal_odds_difference = np.max(np.maximum(np.abs(fpr_group1 - fpr_group2), np.abs(tpr_group1 - tpr_group2)))
    #     print(f"Equal Odds Difference between Group {group1} and Group {group2} is {equal_odds_difference}")

    ##### Equal Odds - End

    

    # Calculate F1 score
    from sklearn.metrics import f1_score
    print("-------------- F1 Score --------------")
    score = f1_score(true_labels2, pred_labels2, average=None)
    all_f1.append(score)
    print("F1 score: ", score)
    lowest = min(score)
    lowest_idx = np.argmin(score) #Based on F1 score
    highest = max(score)
    converged = False
    if (highest - lowest) < 0.05:
        print("Converged for 0.05")
        #converged = True 
    print("Lowest F1 score: ", lowest)
    print("Lowest label: ", )

    # for roc_auc_score use pred_probs (not pred_labels)
    roc_auc_ovr = roc_auc_score(y_ethnicity_onehot, pred_probs, multi_class='ovr', average=None)

    print("ROC score: ", roc_auc_ovr)
    ROC_score.append(roc_auc_ovr)
    print("Total ROC: ", ROC_score)
    print("All F1: ", all_f1)
    
    print(len(pred_labels))
    print(len(true_labels))
    correct0 = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    #correct4 = 0
    false0 = 0
    false1 = 0
    false2 = 0
    false3 = 0
    #false4 = 0

    for i in range(len(true_labels)):
        if true_labels[i] == pred_labels[i]:
            if true_labels[i] == 0:
                correct0 = correct0 + 1
            if true_labels[i] == 1:
                correct1 = correct1 + 1
            if true_labels[i] == 2:
                correct2 = correct2 + 1
            if true_labels[i] == 3:
                correct3 = correct3 + 1
            # if true_labels[i] == 4:
            #     correct4 = correct4 + 1
        else:
            if true_labels[i] == '0':
                false0 = false0 + 1
            if true_labels[i] == '1':
                false1 = false1 + 1
            if true_labels[i] == '2':
                false2 = false2 + 1
            if true_labels[i] == '3':
                false3 = false3 + 1
            # if true_labels[i] == '4':
            #     false4 = false4 + 1

    #print("correct0: ", correct0)
    #print("correct1: ", correct1)
    #print("correct2: ", correct2)
    #print("correct3: ", correct3)
    #print("correct4: ", correct4)
    #print("false0: ", false0)
    #print("false1: ", false1)
    #print("false2: ", false2)

    #print("true_labels.count(2): ", true_labels.count(2))
    #print("true_labels.count(3): ", true_labels.count(3))
    #print("true_labels.count(4): ", true_labels.count(4))

    #print("pred_labels.count(2): ", pred_labels.count(2))
    #print("pred_labels.count(3): ", pred_labels.count(3))
    #print("pred_labels.count(4): ", pred_labels.count(4))

    print("Pred White correct: ", (correct0 / true_labels.count(0)) * 100)
    print("Pred Black correct: ", (correct1 / true_labels.count(1)) * 100)
    print("Pred Asian correct: ", (correct2 / true_labels.count(2)) * 100)
    print("Pred Indian correct: ", (correct3 / true_labels.count(3)) * 100)
    #print("Pred Others correct: ", (correct4 / true_labels.count(4)) * 100)
    acc = ((correct0 / true_labels.count(0)) + (correct1 / true_labels.count(1)) + (correct2 / true_labels.count(2)) + (correct3 / true_labels.count(3))) / 4
    print("accuracy: ", acc)
    all_acc.append(acc)


    #images_names = np.array(["White", "Black", "Asian", "Indian"])
    
    cm = confusion_matrix(true_labels, pred_labels)
    measures = np.zeros((4, len(cm)))

    # Calculate the measures for each class
    for i in range(len(cm)):
        measures[0, i] = cm[i, i]  # TP is diagonal
        measures[1, i] = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]  # TN 
        measures[2, i] = np.sum(cm[:, i]) - cm[i, i]  # FP
        measures[3, i] = np.sum(cm[i, :]) - cm[i, i]  # FN

    # Create a DataFrame for easier viewing
    measures_df = pd.DataFrame(measures, columns=[f'Class {i}' for i in range(len(cm))], 
                            index=['TP', 'TN', 'FP', 'FN'])

    print(measures_df)
    #print(measures[0,0])

    ############ DI
    indian_protected = measures[0][3], measures[2][3], measures[1][3], measures[3][3]  # TP, FP, TN, FN for Indian group
    unprotected_total = [sum(measures[i][j] for i in range(4) if i != 3) for j in range(4)]

    di_total = calculate_disparate_impact(*indian_protected, *unprotected_total)
    all_di.append(di_total)
    print("Total Disparate Impact (DI) for all unprotected groups compared to Indian group:", di_total)
    ############ DI

    eq_odds_diffs, avg_eq_odds_dif = eq_odds(measures_df, cm)
    worst_eq = max(eq_odds_diffs)
    worst_idx = np.argmax(eq_odds_diffs)
    print("Worst eq is ", worst_eq)
    print("Worst eq index is ", worst_idx)
    print("Average eq odds: ", avg_eq_odds_dif)
    all_avgeqodds.append(avg_eq_odds_dif)
    all_eqodds.append(eq_odds_diffs)
    print("All avg Eq Odds: ", all_avgeqodds)
    print("All Eq Odds: ", all_eqodds)
    lowest_idx = worst_idx   # if eqodds is to be used
     
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=images_names)
    
    #disp.plot(cmap=plt.cm.Blues)
    if plot:
        if testset == 1:
            plt.title("Confusion Matrix on FairFace testset")
        elif testset == 2:
            plt.title("Confusion Matrix on UTKFace testset")
        else:
            print("Error encountered")
            exit(1)
        plt.show()

    del image_paths
    del age_labels
    del gender_labels
    del ethnicity_labels
    del indian_protected
    del unprotected_total
    del score
    gc.collect()

    return lowest_idx, converged, all_eqodds, all_acc, all_f1, all_di

def create_model():
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    facenet_model = FaceNet()
    # Freeze layers
    for layer in facenet_model.model.layers[:-10]:
        layer.trainable = False

    input_tensor = Input(shape=input_shape)
    facenet_output = facenet_model.model(input_tensor)
    flatten_layer = Flatten()(facenet_output)
    dense_layer = Dense(256, activation='relu')(flatten_layer)
    dense_layer2 = Dense(128, activation='relu')(dense_layer)
    batch_norm_layer = BatchNormalization()(dense_layer2)
    output_layer = Dense(4, activation='softmax', name='ethnicity_output')(batch_norm_layer)

    model = tf.keras.Model(inputs=input_tensor, outputs=output_layer)

    # Compile the model with a tunable learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def extract_features(images):
    features = []
    for image in images:
        img = tf.keras.utils.load_img(image, color_mode='rgb', target_size=(IMG_SIZE, IMG_SIZE))
        img = np.array(img, dtype=np.uint8)  # Convert to numpy array with the right data type
        features.append(img)

    features = np.array(features)
    del img # Free up memory
    gc.collect()
    return features

