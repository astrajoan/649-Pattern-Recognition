# -*- coding: utf-8 -*-
"""


@author: Ziqi Zhao
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image
import time

global images                    # array of 2499 images, each a 19 x 19 array
global labels                    # array of 2499 labels 1 (face) and -1 (background)
global images_test
global labels_test
global integral_images           # array of 2499 19 x 19 image integrals
global integral_images_test
global features                  # a list of features
global FaceBkgrd_Label_tuple     # tuple of [images, labels]
global FaceBkgrd_Label_tuple_test
global StrongLearner             # Final AdaBoost-ed weak learner set
global THETA                     
global Alphas                    # the alphas for the final weak learner set (generated from AdaBoost)
global feat_list_2h 
global feat_list_2v 
global feat_list_3h 
global feat_list_3v
global feat_list_4
global feat_list_total
global feat_2h
global feat_2v
global feat_3h
global feat_3v
global feat_4
global feat_2h_test
global feat_2v_test
global feat_3h_test
global feat_3v_test
global feat_4_test
global features_test
global FParray
global FNarray
global W



def load_files(filename):
   temp_image = Image.open(filename).convert('L')  #Converts to grayscale
   return np.array(temp_image)


def extract_images(directory):
    temp_array = [load_files(directory + '/'+ filename) for filename in os.listdir(directory)]
    return(temp_array)

def extract_labels(directory):
    if (directory == '/Users/Astra/VJfaces/dataset/trainset/faces'):
        temp_array = [1 for filename in os.listdir(directory)]
    else:
        temp_array = [-1 for filename in os.listdir(directory)]
    return(temp_array)

def extract_labels_test(directory):
    if (directory == '/Users/Astra/VJfaces/dataset/testset/faces'):
        temp_array = [1 for filename in os.listdir(directory)]
    else:
        temp_array = [-1 for filename in os.listdir(directory)]
    return(temp_array)

def load_images(faces_dir, background_dir):
    global images
    global labels
    
    faces = extract_images(faces_dir)
    background = extract_images(background_dir)
    face_labels = extract_labels(faces_dir)
    background_labels = extract_labels(background_dir)
    images = np.concatenate((faces, background),axis=0)
    labels = np.concatenate((face_labels, background_labels),axis=0)
    return(images,labels)
    
    
def load_images_test(faces_dir, background_dir):
    global images_test
    global labels_test
    
    faces = extract_images(faces_dir)
    background = extract_images(background_dir)
    face_labels = extract_labels_test(faces_dir)
    background_labels = extract_labels_test(background_dir)
    images_test = np.concatenate((faces, background),axis=0)
    labels_test = np.concatenate((face_labels, background_labels),axis=0)
    return(images_test,labels_test)

def compute_integral_image(imgs):
   l = len(imgs)
   newImgs = [np.cumsum(np.cumsum(imgs[i], axis=0), axis=1) for i in range(l)]
   finalArr = np.array(newImgs).astype(int)
   return finalArr

# Five types of features
def feature_list_2h(n,stride):
    feat_list = []
    for height in range(1,8,1):
        for width in range(2,8,2):
            for x in range(0,n-height,stride):
                for y in range(0,n-width,stride):
                    w = width // 2
                    top1 = (x, y)
                    top2 = (x, y + w)
                    top3 = (x, y + width)
                    bot1 = (x + height, y)
                    bot2 = (x + height, y + w)
                    bot3 = (x + height, y + width)
                    feat_list.append([top1, top2, top3, bot1, bot2, bot3])
    return feat_list


def feature_list_2v(n,stride):
    feat_list = []
    for height in range(2,8,2):
        for width in range(1,8,1):
            for x in range(0,n-height,stride):
                for y in range(0,n-width,stride):
                    h = height // 2
                    left1  = (x, y)
                    left2 = (x + h, y)
                    left3 = (x + height, y)
                    right1 = (x, y + width)
                    right2 = (x + h, y + width)
                    right3 = (x + height, y + width)
                    feat_list.append([left1, left2, left3, right1, right2, right3])
    return feat_list


def feature_list_3h(n,stride):
    feat_list = []
    for height in range(1,8,1):
        for width in range(3,8,3):
            for x in range(0,n-height,stride):
                for y in range(0,n-width,stride):
                    w = width // 3
                    top1  = (x, y)
                    top2 = (x, y + w)
                    top3 = (x, y + 2 * w)
                    top4 = (x, y + width)
                    bot1 = (x + height, y)
                    bot2 = (x + height, y + w)
                    bot3 = (x + height, y + 2 * w)
                    bot4 = (x + height, y + width)
                    feat_list.append([top1, top2, top3, top4, bot1, bot2, bot3, bot4])
    return feat_list


def feature_list_3v(n,stride):
    feat_list = []
    for height in range(3,8,3):
        for width in range(1,8,1):
            for x in range(0,n-height,stride):
                for y in range(0,n-width,stride):
                    h = height // 3
                    left1  = (x, y)
                    left2 = (x + h, y)
                    left3 = (x + 2 * h, y)
                    left4 = (x + height, y)
                    right1 = (x, y + width)
                    right2 = (x + h, y + width)
                    right3 = (x + 2 * h, y + width)
                    right4 = (x + height, y + width)
                    feat_list.append([left1, left2, left3, left4, right1, right2, right3, right4])
    return feat_list


def feature_list_4(n,stride):
    feat_list = []
    for height in range(2,8,2):
        for width in range(2,8,2):
            for x in range(0,n-height,stride):
                for y in range(0,n-width,stride):
                    w = width // 2
                    h = height // 2
                    top1 = (x, y)
                    top2 = (x, y + w)
                    top3 = (x, y + width)
                    mid1 = (x + h, y)
                    mid2 = (x + h, y + w)
                    mid3 = (x + h, y + width)
                    bot1 = (x + height, y)
                    bot2 = (x + height, y + w)
                    bot3 = (x + height, y + width)
                    feat_list.append([top1, top2, top3, mid1, mid2, mid3, bot1, bot2, bot3])
    return feat_list


# Five types of features on integral images
def compu_feature_2h(int_img):
    global feat_list_2h
    feat_value = []
    for feat_index in range(len(feat_list_2h)):
        feat_value_interm = []
        top1, top2, top3, bot1, bot2, bot3 = feat_list_2h[feat_index]
        for img_index in range(len(int_img)):
            image = int_img[img_index]
            Left =  image[top1[0]][top1[1]] + image[bot2[0]][bot2[1]]- image[bot1[0]][bot1[1]] - image[top2[0]][top2[1]]
            Right = image[top2[0]][top2[1]] + image[bot3[0]][bot3[1]] - image[bot2[0]][bot2[1]] - image[top3[0]][top3[1]]
            feat_value_interm.append(Left-Right)
        feat_value.append(feat_value_interm)
    return feat_value


def compu_feature_2v(int_img):
    global feat_list_2v
    feat_value = []
    for feat_index in range(len(feat_list_2v)):
        feat_value_interm = []
        left1, left2, left3, right1, right2, right3 = feat_list_2v[feat_index]
        for img_index in range(len(int_img)):
            image = int_img[img_index]
            Top =  image[left1[0]][left1[1]] + image[right2[0]][right2[1]] - image[left2[0]][left2[1]] - image[right1[0]][right1[1]]
            Bottom = image[left2[0]][left2[1]] + image[right3[0]][right3[1]] - image[left3[0]][left3[1]] - image[right2[0]][right2[1]]
            feat_value_interm.append(Top - Bottom)
        feat_value.append(feat_value_interm)
    return feat_value


def compu_feature_3h(int_img):
    global feat_list_3h
    feat_value = []
    for feat_index in range(len(feat_list_3h)):
        feat_value_interm = []
        top1, top2, top3, top4, bot1, bot2, bot3, bot4 = feat_list_3h[feat_index]
        for img_index in range(len(int_img)):
            image = int_img[img_index]
            Left = image[top1[0]][top1[1]] + image[bot2[0]][bot2[1]] - image[bot1[0]][bot1[1]] - image[top2[0]][top2[1]]
            Mid =  image[top2[0]][top2[1]] + image[bot3[0]][bot3[1]] - image[bot2[0]][bot2[1]] - image[top3[0]][top3[1]]
            Right = image[top3[0]][top3[1]] + image[bot4[0]][bot4[1]] - image[bot3[0]][bot3[1]] - image[top4[0]][top4[1]]
            feat_value_interm.append(Left + Right - Mid)
        feat_value.append(feat_value_interm)
    return feat_value


def compu_feature_3v(int_img):
    global feat_list_3v
    feat_value = []
    for feat_index in range(len(feat_list_3v)):
        feat_value_interm = []
        left1, left2, left3, left4, right1, right2, right3, right4 = feat_list_3v[feat_index]
        for img_index in range(len(int_img)):
            image = int_img[img_index]
            Top =  image[left1[0]][left1[1]] + image[right2[0]][right2[1]] - image[left2[0]][left2[1]] - image[right1[0]][right1[1]]
            Mid =  image[left2[0]][left2[1]] + image[right3[0]][right3[1]] - image[left3[0]][left3[1]] - image[right2[0]][right2[1]]
            Bottom =  image[left3[0]][left3[1]] + image[right4[0]][right4[1]] - image[left4[0]][left4[1]] - image[right3[0]][right3[1]]
            feat_value_interm.append(Top + Bottom - Mid)
        feat_value.append(feat_value_interm)
    return feat_value


def compu_feature_4(int_img):
    global feat_list_4
    feat_value = []
    for feat_index in range(len(feat_list_4)):
        feat_value_interm = []
        top1, top2, top3, mid1, mid2, mid3, bot1, bot2, bot3 = feat_list_4[feat_index]
        for img_index in range(len(int_img)):
            image = int_img[img_index]
            Left_Top =  image[top1[0]][top1[1]] + image[mid2[0]][mid2[1]] - image[mid1[0]][mid1[1]] - image[top2[0]][top2[1]]
            Right_Top =  image[top2[0]][top2[1]] + image[mid3[0]][mid3[1]] - image[mid2[0]][mid2[1]] - image[top3[0]][top3[1]]
            Left_Bottom =  image[mid1[0]][mid1[1]] + image[bot2[0]][bot2[1]] - image[bot1[0]][bot1[1]] - image[mid2[0]][mid2[1]]
            Right_Bottom =  image[mid2[0]][mid2[1]] + image[bot3[0]][bot3[1]] - image[bot2[0]][bot2[1]] - image[mid3[0]][mid3[1]]
            feat_value_interm.append(Left_Top + Right_Bottom - Left_Bottom - Right_Top)
        feat_value.append(feat_value_interm)
    return feat_value

def optimal_p_theta(weights, f):
    global features
    global labels
    total_pos, total_neg = 0, 0
    for w, label in zip(weights, labels):
        if label == 1:
            total_pos += w
        else:
            total_neg += w

    applied_feature = sorted(zip(weights, features[f], labels), key=lambda x: x[1])

    pos_exist, neg_exist = 0, 0
    pos_weights, neg_weights = 0, 0
    min_error, best_theta, best_p = float('inf'), None, None
    for w, f, label in applied_feature:
        error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
        if error < min_error:
            min_error = error
            best_theta = f
            best_p = 1 if pos_exist > neg_exist else -1

        if label == 1:
            pos_exist += 1
            pos_weights += w
        else:
            neg_exist += 1
            neg_weights += w

    return (best_theta, best_p)

# Calculate the prediction of feature #f on all images
def weaklearner_value(int_img, p, theta, f):
    global features
    
    N = len(int_img)
    predictions = [0 for x in range(N)]
    
    for i in range(N):
        raw_prediction = p * (features[f][i] - theta)
        if (raw_prediction >= 0):
            predictions[i] = 1
        else:
            predictions[i] = -1
    return predictions


# Calculate the error rate of feature #f
def error_rate(int_img, weights, f, p, theta):
    global labels
    
    N = len(int_img)
    
    predictions = weaklearner_value(int_img, p, theta, f)
    weighted_error = 0
    
    for i in range(N):
        if (predictions[i] != labels[i]):
            factor = 1
        else:
            factor = 0
        weighted_error += weights[i] * factor
#    print("--- error_rate feature: ", f, " %s seconds ---" % (time.time() - start_time))
# =============================================================================
#     # False Positive
#     for i in range(N):
#         if (labels[i] == -1):
#             if (predictions[i] == 1):
#                 weighted_error += weights[i]
#     
#     
#     # False Negative
#     for i in range(N):
#         if (labels[i] == 1):
#             if (predictions[i] == -1):
#                 weighted_error += weights[i]
# =============================================================================
    return weighted_error

def optimal_weaklearner(int_img, weights):
    global features
    
    error_arr = []
    
    for f in range(len(features)):
        if (f % 300 == 0):
            print('         optimal_weaklearner: starting feature #', f, "of",len(features))
        theta,p = optimal_p_theta(weights, f)                        
        error_arr.append(error_rate(int_img, weights, f, p, theta))  # Calculate the error rate of each feature
    
#    opt = np.argwhere (x == np.min(error_arr))
    opt = np.argmin(error_arr)
    theta, p = optimal_p_theta(weights, opt)
    return(opt, p, theta)

# Update the weights 
def update_weights(weights, error_rate, y_pred, y_true): 
    LW = len(weights)
    updated_weights = np.empty(shape=LW)
    for i in range(LW):
        if (error_rate < 0.05):
            Z_t = 1
            Alpha_t = 0
        else:
            Z_t = 2 * np.sqrt(error_rate*(1 - error_rate))
            Alpha_t = 0.5 * np.log((1 - error_rate)/error_rate)
        update_factor = np.exp(-1 * Alpha_t * y_true[i] * y_pred[i])
        updated_weights[i] = (update_factor * weights[i])/Z_t
    return updated_weights   


# Use the weak learners to compute the "strong learner" prediction
def strongPrediction(weakLearners,alphas, image_idx, feat):
    A = len(alphas)         # No. of weak learners
    strong_pred = 0
    for i in range(A):
        f = int(weakLearners[i][0])
        p = int(weakLearners[i][1])
        theta = weakLearners[i][2] 
        H = feat[f][image_idx]
        value = p * (H - theta)
        strong_pred += value * alphas[i]
    return strong_pred

def compute_FPR(weaklearners,alphas, y_true, int_img, feat):  
    N = len(int_img)
    FPcount = 0
    for i in range(N):
        if (y_true[i] == -1):
            prediction = strongPrediction(weaklearners,alphas,i,feat)
            if (prediction >= 0.5 * sum(alphas)):
                FPcount += 1
    return(FPcount/N)

def compute_FNR(weaklearners,alphas, y_true, int_img, feat):  
    N = len(int_img)
    FNcount = 0
    for i in range(N):
        if (y_true[i] == 1):
            prediction = strongPrediction(weaklearners,alphas,i,feat)
            if (prediction < 0.5 * sum(alphas)):
                FNcount += 1
    return(FNcount/N)
    
def Accuracy(weaklearners,alphas, y_true, int_img, feat):
    N = len(int_img)
    errcount = 0
    
    for i in range(N):
        prediction = strongPrediction(weaklearners, alphas, i, feat)
        if (prediction >= 0.5 * sum(alphas)):
            predict = 1
        else:
            predict = -1
        if (predict != y_true[i]):
            errcount += 1
    return (errcount/N)
    
    
def feature_train_accuracy(int_img, top_f):
    global labels
    global StrongLearner
    N = len(int_img)
    f = StrongLearner[top_f][0]
    p = StrongLearner[top_f][1]
    theta = StrongLearner[top_f][2]
    errorcount = 0
    
    predictions = weaklearner_value(int_img, p, theta, int(f))
    for i in range(N):
        if (predictions[i] != labels[i]):
            errorcount += 1
    err = errorcount / N
    return (1-err)
            
    
def AdaBoost(int_img, y_true):
    global features
    global FParray
    global FNarray
    global strides
    global W
    
    W = 10
    N = len(int_img)
    weights = [1/N for i in range(N)]           # initial equal weights adding up to 1
    weakLearners = np.empty(shape=(W,3))        
    alphas = np.empty(shape=W)   
    FPrate = 1                                  
    FNrate = 1                                  
    iterations = 0
    FParray = []                               
    FNarray = []  
    accuracy_array = []
                              
    print("START AdaBoost. LONG iterations!!! Alerts for every 300 features.")
    while (iterations < W):
        print("Starting AdaBoost iteration#",iterations + 1," - calling optimal_weaklearner .....")
        feat_idx,p,theta = optimal_weaklearner(int_img, weights)
        predictions = weaklearner_value(int_img, int(p), theta, int(feat_idx))
        weighted_error = error_rate(int_img, weights, int(feat_idx) ,int(p), theta)
        alpha = np.log(((1 - weighted_error)/(weighted_error)))
        weights = update_weights(weights,weighted_error,predictions,y_true)
        alphas[iterations] = alpha
        print("Optimal learner index:",feat_idx," - p / theta are (respectively) ",p, "/", theta)
        weakLearners[iterations][0] = feat_idx
        weakLearners[iterations][1] = p
        weakLearners[iterations][2] = theta
        FPrate = compute_FPR(weakLearners[:iterations + 1],alphas[:iterations + 1], y_true, int_img, features)
        FParray.append(FPrate)
        FNrate = compute_FNR(weakLearners[:iterations + 1],alphas[:iterations + 1], y_true, int_img, features)
        FNarray.append(FNrate)
        accuracy = 1 - Accuracy(weakLearners[:iterations + 1],alphas[:iterations + 1], y_true, int_img, features)
        accuracy_array.append(accuracy)
        print("FALSE NEGATIVE/FALSE POSITIVE rates, accuracy after iteration #", iterations + 1, "are:",FNrate,"/", FPrate, "/", accuracy, "      Weights have been updated\n")
        iterations += 1
        
    # Plot the false negative and false positive rates
    print ("Plots for FALSE NEGATIVES (BLUE), FALSE POSITIVES (ORANGE) and accuracy (Green) are shown below (x-axis = iteration #):")
    fig_rates=plt.figure()
    plt.plot(FNarray)
    plt.plot(FParray)
    plt.plot(accuracy_array)
    fig_rates.suptitle('FPR, FNR and accuracy for '+str(W)+' rounds-'+str(strides)+' strides')
    fig_rates.savefig('round'+str(W)+'-stride'+str(strides)+'.eps',format='eps')
    
    # Get the minimum strong prediction  
    minval = 100000000      
    for i in range(N):
        if(y_true[i] == 1):     # This is a face
            strongval = strongPrediction(weakLearners[:iterations],alphas[:iterations], i, features)
            if(strongval < minval):
                minval = strongval
    return (weakLearners[:iterations], alphas[:iterations],minval)

######################### Main program starts here #####################################################    
def main():
    global integral_images
    global integral_images_test
    global FaceBkgrd_Label_tuple
    global FaceBkgrd_Label_tuple_test
    global features
    global labels
    global labels_test
    global StrongLearner             # Final AdaBoost-ed weak learner set
    global THETA                     
    global Alphas                    # the alphas for the final weak learner set (generated from AdaBoost)
    global feat_list_2h 
    global feat_list_2v 
    global feat_list_3h 
    global feat_list_3v
    global feat_list_4
    global feat_list_total
    global feat_2h
    global feat_2v
    global feat_3h
    global feat_3v
    global feat_4
    global feat_2h_test
    global feat_2v_test
    global feat_3h_test
    global feat_3v_test
    global feat_4_test
    global features_test
    global strides
    global W
    
    
    print("START loading TRAINING faces and backgrounds and form label array (+1 and -1) ....")
    FaceBkgrd_Label_tuple = load_images('/Users/Astra/VJfaces/dataset/trainset/faces','/Users/Astra/VJfaces/dataset/trainset/non-faces')
    print("FINISHED loading faces/backgrounds and labels into arrays\n")
    time.sleep(2)
    
    print("START computing the integral image array .....")
    integral_images = compute_integral_image(FaceBkgrd_Label_tuple[0])
    print("FINISHED computing the integral image array\n")
    time.sleep(2)
     
    
    print("START forming feature list and compute feature values on TRAINING IMAGES.....")
    strides=1
    feat_list_2h = feature_list_2h(19,strides)
    feat_list_2v = feature_list_2v(19,strides)
    feat_list_3h = feature_list_3h(19,strides)
    feat_list_3v = feature_list_3v(19,strides)
    feat_list_4 = feature_list_4(19,strides)
    feat_list_total = []
    feat_list_total = feat_list_2h + feat_list_2v + feat_list_3h + feat_list_3v + feat_list_4
    
    feat_2h = compu_feature_2h(integral_images)
    feat_2v = compu_feature_2v(integral_images)
    feat_3h = compu_feature_3h(integral_images)
    feat_3v = compu_feature_3v(integral_images)
    feat_4 = compu_feature_4(integral_images)
    features = []
    features = feat_2h + feat_2v + feat_3h + feat_3v + feat_4
    print("FINISHED forming the feature list\n")
    time.sleep(2)
    
    ############# START the TRAINING part of the program ############################
    StrongLearner, Alphas, THETA = AdaBoost(integral_images, labels)
    print("AdaBoost HAS COMPLETED!\n")
    for i in range(W):
        print("Top feature: ",i+1, "training accuracy: ", feature_train_accuracy(integral_images, i))
    ############# END the TRAINING part of the program ##############################
    
    ############# START the TESTING part of the program ############################
    print("START loading TESTING faces and backgrounds and form label array (+1 and -1) ....")
    FaceBkgrd_Label_tuple_test = load_images_test('/Users/Astra/VJfaces/dataset/testset/faces','/Users/Astra/VJfaces/dataset/testset/non-faces')
    print("FINISHED loading faces/backgrounds and labels into arrays\n")
    time.sleep(2)
    
    print("START computing the integral image array .....")
    integral_images_test = compute_integral_image(FaceBkgrd_Label_tuple_test[0])
    print("FINISHED computing the integral image array\n")
    time.sleep(2)
    
    
    print("START forming feature list and compute feature values on TESTING IMAGES.....")
    feat_2h_test = compu_feature_2h(integral_images_test)
    feat_2v_test = compu_feature_2v(integral_images_test)
    feat_3h_test = compu_feature_3h(integral_images_test)
    feat_3v_test = compu_feature_3v(integral_images_test)
    feat_4_test = compu_feature_4(integral_images_test)
    features_test = []
    features_test = feat_2h_test + feat_2v_test + feat_3h_test + feat_3v_test + feat_4_test
    print("FINISHED forming the feature list\n")
    time.sleep(2)
    
    
    FP_test = []
    FN_test = []
    accuracy_test = []
    for i in range(W):
        FP = compute_FPR(StrongLearner[:i+1], Alphas[:i+1], labels_test, integral_images_test, features_test)
        FN = compute_FNR(StrongLearner[:i+1], Alphas[:i+1], labels_test, integral_images_test, features_test)
        accuracy = 1- Accuracy(StrongLearner[:i+1], Alphas[:i+1], labels_test, integral_images_test, features_test)
        FP_test.append(FP)
        FN_test.append(FN)
        accuracy_test.append(accuracy)
        print ("TESTING after ", i, "iterations\n", "False Positive: ", FP, "and False Negative: ", FN, "\n accuracy:", accuracy )
    
    
    print ("Plots for FALSE NEGATIVES (BLUE), FALSE POSITIVES (ORANGE) and Accuracy (Green) for TESTING are shown below (x-axis = iteration #):")
    fig_rates_test=plt.figure()
    plt.plot(FN_test)
    plt.plot(FP_test)
    plt.plot(accuracy_test)
    fig_rates_test.suptitle('FPR, FNR and accuracy for '+str(W)+' rounds-'+str(strides)+' strides'+' TESTING')
    fig_rates_test.savefig('round'+str(W)+'-stride'+str(strides)+'-TESTING'+'.eps',format='eps')
    
    
    ############# END the TESTING part of the program ############################


main()


