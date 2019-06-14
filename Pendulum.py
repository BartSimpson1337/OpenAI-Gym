import gym
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

steps = 3500
sample = 25000
games = 10
score_requirement = -100
reward_requirement = -15.2736044


env = gym.make('Pendulum-v0')
env.reset()

def ReadData():
    global steps
    global sample
    global score_requirement
    global reward_requirement

    rows = []
    for i in range(sample):
        score = 0
        game_memory = []
        prev_observation = []
        for j in range(steps):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action]) 
            prev_observation = observation

            if reward <= reward_requirement:
                score += reward
            if done:
                break

            
        if score <= score_requirement:
            print("random game score:", score)
            for data in game_memory:
                columns = []
                for i in range(len(data[0])):
                    columns.append(data[0][i]) #Features
                columns.append(data[1][0]) #Class
                
                rows.append(columns)
        env.reset()

    data = pd.DataFrame(rows)
    data.columns = ["Cosine", "Sine", "Theta", "Action"]
    return data


def TrainingData(dataset):
    x_train, feature_scaler = Normalization(dataset.iloc[:, :-1]) #performing normalization in feature columns of the dataset
    y_train, class_scaler = Normalization(pd.DataFrame(np.reshape(np.array(dataset.iloc[:, -1]), (-1, 1)))) #performing normalization in class column of the dataset
    y_train.columns = ["Action"]
    return x_train, y_train.iloc[:, 0], feature_scaler, class_scaler



def Normalization(rows):
    scaler = MinMaxScaler()
    df = scaler.fit_transform(rows)
    dataset = pd.DataFrame(df, columns=rows.columns[:]) 
    return dataset, scaler


def Predictor():

    cos_var = tf.feature_column.numeric_column('Cosine')
    sin_var = tf.feature_column.numeric_column('Sine')
    theta_var = tf.feature_column.numeric_column('Theta')

    features = [cos_var,sin_var,theta_var]
    return tf.estimator.DNNRegressor(hidden_units=[10 for i in range(5)],
                                      feature_columns=features, 
                                      activation_fn=tf.nn.relu,
                                      model_dir="/tmp/Pendulum",
                                      config=tf.contrib.learn.RunConfig(
                                            save_checkpoints_steps=250,
                                            save_checkpoints_secs=None,
                                            save_summary_steps=500))


def Model(predictor, x_train, y_train):

    training_input_fn = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train, batch_size=len(x_train), shuffle=True, num_epochs=3)
    model = predictor.train(input_fn=training_input_fn, steps=300)
    return model

def Predict(model, x_test):
    x_test = pd.DataFrame(x_test)
    x_test.columns = ["Cosine", "Sine", "Theta"]

    pre_input_fn = tf.estimator.inputs.pandas_input_fn(x=x_test, shuffle=False)
    predictions = list(model.predict(input_fn=pre_input_fn))

    y_pred = []
    for pred in predictions:
        y_pred.append(pred['predictions'][0])

    return y_pred


def Run(dataset):
    global games
    global steps


    x_train, y_train, feature_scaler, class_scaler = TrainingData(dataset)

    model = Model(Predictor(), x_train, y_train) #Creating a DNN model
    
    scores = []
    for game in range(games):
        score = 0
        testing = []
        env.reset()
        for step in range(steps):
            env.render()

            action = None
            if len(testing) == 0: #To run 
                action = env.action_space.sample()
            else:
                prediction = Predict(model, feature_scaler.transform(pd.DataFrame(testing).T))#performing a normalization in test row, just before predict it.
                action = class_scaler.inverse_transform(np.reshape(prediction, (-1, 1)))[0]
                print("Iteraction:", step,"Action: ", action[0])
                
            observation, reward, done, info = env.step(action)

            columns = []
            for i in range(len(observation)):
                columns.append(observation[i])

            testing = columns
            score += reward
        scores.append(score)
    print('Average score: ', np.mean(scores))
def main():
    dataset = ReadData()
    Run(dataset)

main()



