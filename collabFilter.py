import sys
import pdb
import json
import math
import operator
import numpy as np
import pandas as pd

class User:
    def __init__(self, uid, avg_rating, ratings):
        self.uid = uid
        self.avg_rating = avg_rating
        self.ratings = ratings
        self.predictions = {}
        # {dish_id : rating}
        self.combined = ratings
users_list = []

# with open('data/myfile.json') as f:
with open('data/user_ratings_train.json') as f:
    train_data = json.load(f)
    num_users_train = len(train_data)

with open('data/user_ratings_test.json') as f:
    test_data = json.load(f)
    num_users_test = len(test_data)
    test_users = set(list(test_data.keys()))
    test_users = {int(x) for x in test_users}

dishes_df = pd.read_csv('data/dishes.csv')
num_dishes = len(dishes_df.index)
# 2d array of similarity
user_weights = [[None]*num_users_train for x in range(num_users_train)]

relevant_dishes = [set() for i in range(num_users_train)]
def compute_relevant():
    num_rel = 0
    # find relevant in training data
    for user in train_data:
        uid = int(user)
        if uid in test_users:
            for dish in train_data[user]:
                if dish[1] >= 3:
                    relevant_dishes[uid].add(dish[0])
                    num_rel += 1
    # test data
    for user in test_data:
        uid = int(user)
        for dish in test_data[user]:
            if dish[1] >= 3:
                relevant_dishes[uid].add(dish[0])
                num_rel += 1
    return num_rel

# used to create list of User objects with relevant info
def user_info():
    # with open('data/myfile.json') as f:
    # user - key
    for user in train_data:
        # {dish : rating}
        ratings_vector = {}
        uid = int(user)
        
        # calculate average rating of user
        num_ratings = 0
        sum_ratings = 0
        for dish_info in train_data[user]:
            ratings_vector[dish_info[0]] = dish_info[1] 
            sum_ratings += dish_info[1]
            num_ratings += 1
        avg_rating = sum_ratings/float(num_ratings)
        
        # if uid == 0:
        #     for v in ratings_vector.values():
        #         print(f"{v}, ", end="")

        # if uid == 400:
        #     print(f"Average: {avg_rating}")

        # add new user to list
        new_user = User(uid, avg_rating, ratings_vector)
        users_list.append(new_user)


def calc_weights():
    for i in range(len(users_list)):
        curr_user = users_list[i]
        for j in range(i+1, len(users_list)):
            other_user = users_list[j]
            if user_weights[curr_user.uid][other_user.uid] is None and user_weights[other_user.uid][curr_user.uid] is None:
                weight = compute_pearson(curr_user, other_user)
                user_weights[curr_user.uid][other_user.uid] = weight
                user_weights[other_user.uid][curr_user.uid] = weight


# Pearson correlation coefficient similarity 
def compute_pearson(curr_user, other_user):
    numerator = 0
    first_denom = 0
    second_denom = 0
    for r in curr_user.ratings:
        if r in other_user.ratings:
            first_value = curr_user.ratings[r] - curr_user.avg_rating
            second_value = other_user.ratings[r] - other_user.avg_rating
            numerator += first_value * second_value
            first_denom += pow(first_value, 2)
            second_denom += pow(second_value, 2)
    weight = numerator / (math.sqrt(first_denom) * math.sqrt(second_denom))
    return weight

# Fill up ratings hashmap
def make_predictions(ndishes):
    # iterate through each user, find missing dish ratings
    for curr_user in users_list:
        curr_avg = curr_user.avg_rating
        curr_id = curr_user.uid
        
        # check if user in test set
        if curr_id in test_users:
            # iterate through each possible dish
            for dish_id in range(ndishes):
                # if no rating present
                if dish_id not in curr_user.ratings:
                    numerator = 0
                    denominator = 0
                    # iterate through other users
                    for other_user in users_list:
                        other_id = other_user.uid
                        # if different user and different user contains dish, calculate variables in formula
                        if curr_id != other_id and dish_id in other_user.ratings:
                            numerator += user_weights[curr_id][other_id] * (other_user.ratings[dish_id] - other_user.avg_rating)
                            denominator += abs(user_weights[curr_id][other_id])
                    # calculate prediction
                    prediction = round(curr_avg + numerator/denominator)
                    curr_user.predictions[dish_id] = float(prediction)


def predict_ratings():
    user_info()
    calc_weights()
    make_predictions(num_dishes)


def combine_ratings():
    for user in users_list:
        for k, v in user.predictions.items():
            user.combined[k] = v
        

# precision = num_rel_ret/num_ret
# recall = num_rel_ret/num_rel
def calc_metrics(total_relevant):
    mae = calc_mae() 
    num_ret_10 = 10*num_users_test
    num_ret_20 = 20*num_users_test
    
    # sort and store dish_ids
    num_relret_10 = num_relret_20 = 0
    for i in test_users:
        # sort dishes in users_list[i].combined, store top 10 and 20
        dish_ratings = [(k, v) for k, v in users_list[i].combined.items()]
        
        # if i == 399:
        #     print(f"{len(dish_ratings)}")
        # dish_ratings.sort(key=lambda x: x[0], reverse=True)
        dish_ratings.sort(key=lambda x: x[1], reverse=True)
        top10 = {x[0] for x in dish_ratings[:10]}
        top20 = {x[0] for x in dish_ratings[:20]}

        if i == 399:
            print(f"Top 20: {top20}")
            print(f"Relevant: {relevant_dishes[i]}\n")
            print(f"Relevant retrieved: {relevant_dishes[i].intersection(top20)}")
            # for dish in dish_ratings:
            #     print(f"{dish[0]}={dish[1]}", end=", ")

        num_relret_10 += len(top10.intersection(relevant_dishes[i]))
        num_relret_20 += len(top20.intersection(relevant_dishes[i]))
        
    pat10 = num_relret_10 / num_ret_10 
    pat20 = num_relret_20 / num_ret_20  
    rat10 = num_relret_10 / total_relevant
    rat20 = num_relret_20 / total_relevant

    return mae, pat10, pat20, rat10, rat20
        

def calc_mae():
    deviations = 0
    num = 0
    for user in test_data:
        # list of lists
        ratings = test_data[user]
        for r in ratings:
            prediction = users_list[int(user)].predictions[r[0]]
            deviations += abs(prediction - r[1])
            num += 1
    return deviations / num
        

if __name__ == "__main__":
    num_rel = compute_relevant()
    predict_ratings()
    # pdb.set_trace()
    # print(f"User 399 = {len(users_list[399].ratings)}+{len(users_list[399].predictions)}")
    combine_ratings()
    mae, pat10, pat20, rat10, rat20 = calc_metrics(num_rel)
    # for temp in user_weights:
    #     if 0 in temp:
    #         print("True")
    # weights_400 = user_weights[400]
    # print(f"{user_weights[400][311]}")
    # print(f"User 400 weights: {sorted(weights_400, key=lambda x: (x is None, x))[:10]}")
    # print(f"num_rel = {num_rel}")
    # print(f"{relevant_dishes[400]}")
    print(f"Task 1 MAE: {mae}")
    print(f"Task 2 Precision@10: {pat10}")
    print(f"Task 2 Precision@20: {pat20}")
    print(f"Task 2 Recall@10: {rat10}")
    print(f"Task 2 Recall@20: {rat20}")