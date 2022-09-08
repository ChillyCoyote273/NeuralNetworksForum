from tensorflow import keras
from keras.models import Sequential
from keras import Input
from keras.layers import Dense

import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def main():
	file_name = "C:\\Users\\iamco\\VSCodeProjects\\Python\\NeuralNetworksForum\\Week 3\\weatherAUS.csv"

	df = pd.read_csv(file_name, encoding='utf-8')
	pd.options.display.max_columns=50

	print("1")

	df = df[pd.isnull(df['RainTomorrow']) == False]
	df = df.fillna(df.mean())

	print("2")

	df['RainTodayFlag'] = df['RainToday'].apply(lambda x: 1 if x=='Yes' else 0)
	df['RainTomorrowFlag'] = df['RainTomorrow'].apply(lambda x: 1 if x=='Yes' else 0)

	X = df[['Humidity3pm']]
	y = df['RainTomorrowFlag'].values

	print("3")

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	model = Sequential(name="Model-with-One-Input")

	model.add(Input(shape=(1,), name="Input-Layer"))
	model.add(Dense(2, activation="softplus", name="Hidden-Layer"))
	model.add(Dense(1, activation="sigmoid", name="Output-Layer"))

	model.compile(loss="binary_crossentropy",
		metrics=["Accuracy", "Precision", "Recall"])

	model.fit(X_train, y_train, batch_size=10, epochs=3)


if __name__ == "__main__":
	main()
