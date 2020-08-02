# LSTM for international airline passengers problem with time step regression framing
import numpy
import os
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
import random
import csv
from sklearn.preprocessing import MinMaxScaler
import pywt
import numpy as np


from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets,cross_validation,ensemble
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score


from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from keras.layers import LSTM,GRU
from sklearn.preprocessing import MinMaxScaler
from keras.layers.core import Dense, Activation
from sklearn.metrics import mean_squared_error
from keras import metrics
from sklearn.metrics import r2_score
import sklearn.metrics as metrics
from keras.models import load_model
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import svm
import itertools
from sklearn.metrics import precision_recall_fscore_support as score
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector

from scipy import interpolate
from scipy.fftpack import fft,ifft
from numpy import mean, ptp, var, std
from keras.layers import Bidirectional, CuDNNLSTM



def MAPE(y_true, y_pred):
	"""Mean Absolute Percentage Error
	Calculate the mape.

	# Arguments
		y_true: List/ndarray, ture data.
		y_pred: List/ndarray, predicted data.
	# Returns
		mape: Double, result data for train.
	"""
	y = [x for x in y_true if x > 0]
	y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]
	num = len(y_pred)
	sums = 0
	for i in range(num):
		tmp = abs(y[i] - y_pred[i]) / y[i]
		sums += tmp
	mape = sums * (100 / num)
	return mape


def eva_matrics(y_true, y_pred):
	"""Evaluation
	evaluate the predicted resul.

	# Arguments
		y_true: List/ndarray, ture data.
		y_pred: List/ndarray, predicted data.


	模型效果指标评估
	y_true：真实的数据值
	y_pred：回归模型预测的数据值
	explained_variance_score：解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量
	的方差变化，值越小则说明效果越差。
	mean_absolute_error：平均绝对误差（Mean Absolute Error，MAE），用于评估预测结果和真实数据集的接近程度的程度
	，其其值越小说明拟合效果越好。
	mean_squared_error：均方差（Mean squared error，MSE），该指标计算的是拟合数据和原始数据对应样本点的误差的
	平方和的均值，其值越小说明拟合效果越好。
	r2_score：判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因
	变量的方差变化，值越小则说明效果越差。
	"""

	# mape = MAPE(y_true, y_pred)
	vs = metrics.explained_variance_score(y_true, y_pred)
	mae = metrics.mean_absolute_error(y_true, y_pred)
	mse = metrics.mean_squared_error(y_true, y_pred)
	r2 = metrics.r2_score(y_true, y_pred)
	print('explained_variance_score:%f' % vs, "   :larger is better")
	# print('mape:%f%%' % mape, "   :smaller is better")
	print('mae:%f' % mae,  "   :smaller is better")
	print('mse:%f' % mse, "   :smaller is better")
	print('rmse:%f' % math.sqrt(mse),  "   :smaller is better")
	print('r2:%f' % r2, "   : =1 is better")
	# r2_document[name] = r2

# convert an array of values into a dataset matrix
def create_dataset(dataset, real_test, feature, rapid_feature, look_back ):
	dataX, dataY = [], []

	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		b = feature[i:(i+look_back), 0]

		a = a.tolist()
		b = b.tolist()
		d = []
		for j in range(len(a)):
			d.append(a[j])
			d.append(b[j])
			if j != (look_back-1):
				d.append(0)
			else:
				d.append(rapid_feature[i])

		dataX.append(d)
		dataY.append(real_test[i + look_back: i + look_back+2, 0])
		# dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


def plot_all_data(trainPredict, testPredict, dataset,flag):

	# shift train predictions for plotting
	trainPredictPlot = numpy.empty_like(dataset)
	trainPredictPlot[:, :] = numpy.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

	testPredictPlot = numpy.empty_like(dataset)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1-flag, :] = testPredict

	plt.plot(scaler.inverse_transform(dataset))
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.show()


def plot_testing_result(trainPredict,testPredict,dataset,flag):

	testPredictPlot = numpy.empty_like(dataset)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1-flag, :] = testPredict

	# plot baseline and predictions

	dataset = scaler.inverse_transform(dataset)
	testdata_plot = numpy.empty_like(dataset)
	testdata_plot[:, :] = numpy.nan
	testdata_plot[len(trainPredict)+(look_back*2)+1:len(dataset)-1-flag, :] = dataset[(len(dataset)-len(testPredict)):]

	plt.plot(testdata_plot)
	plt.plot(testPredictPlot)
	plt.show()


def testing(trainX, testX, trainY, testY, dataset, flag):

	lstm = load_model('model/gru_4_32_2000_2step_STR_upgrade_speed.h5')
	file = 'model/gru_4_32_2000_2step_STR_upgrade_speed.png'
	plot_model(lstm, to_file=file, show_shapes=True)

	trainPredict = lstm.predict(trainX)
	testPredict = lstm.predict(testX)

	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])				# true data
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY]) 				# true data

	print(testPredict)
	eva_matrics(testY[0], testPredict[:,0])

	# plot_all_data(trainPredict, testPredict, dataset, flag)
	# plot_testing_result(trainPredict,testPredict,dataset, flag)

def wavelet(dataset):

	data = dataset.tolist()

	new = []
	for i in range(len(data)):
		new.append(data[i])

	index = []
	data = new
	for i in range(len(new)):
		X = float(i)
		# Y = float(ecg[i])
		index.append(X)
		# data.append(Y)

	w=pywt.Wavelet('sym18')                             #选用Daubechies8小波
	maxlev=pywt.dwt_max_level(len(data),w.dec_len)
	threshold=2                                         #Threshold for filtering
	coffs=pywt.wavedec(data,'sym18',level=maxlev)       #将信号进行小波分解
	for i in range(1,len(coffs)):
		coffs[i]=pywt.threshold(coffs[i],threshold*max(coffs[i]))

	datarec = pywt.waverec(coffs,'sym18')               #将信号进行小波重构

	gogogo = []
	for i in datarec:
		i = i.tolist()
		gogogo.append([i])

	# dd = datarec[0].tolist()
	# cc = datarec[1].tolist()
	# aa = []
	# aa.append([dd])
	# aa.append([cc])
	gogogo = numpy.asarray(gogogo)

	return gogogo


def moving_average(x,w):

	y = []
	x = x.tolist()
	for i in x:
		y.append(i[0])

	y = numpy.array(y)
	y = numpy.ravel(y)
	return numpy.convolve(y, numpy.ones(w), 'valid') / w


def creat_label():

	dataframe2 = read_csv('shangzheng.csv', usecols=[7], engine='python')
	tmp2 = dataframe2.values.tolist()
	tmp_list2 = []
	for j in range(len(tmp2)):
		tmp_list2.append(tmp2[j][0])
	tmp_list2 = numpy.array(tmp_list2)
	dataset2 = numpy.reshape(tmp_list2, (tmp_list2.shape[0], 1))

	# print(dataset2)
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset2 = scaler.fit_transform(dataset2)

	return dataset2


def use_gru(trainX,trainY,testX,testY):

	# model = Sequential()
	# model.add(LSTM(64, return_sequences=True, input_shape=(4, 45)))
	# model.add(LSTM(32,return_sequences=True))
	# model.add(LSTM(16,return_sequences=True))
	# model.add(LSTM(8,return_sequences=False))
	# model.add(Dense(8, activation='softmax'))
	# model.add(Dense(1))
	# model.summary()
	# model.compile(loss='mse', optimizer='adam', metrics=['mape']) # metrics=['mape']
	# hist= model.fit(trainX, trainY, epochs=2000, batch_size=32, verbose=1)#


	# model.save('test'+'.h5')
	# print("Done~!")


	lstm = load_model('test'+'.h5')
	testPredict = lstm.predict(testX)
	testPredict = scaler.inverse_transform(testPredict)
	testY = np.reshape(testY,(testY.shape[0],1))
	testY = scaler.inverse_transform(testY)

	testY = testY[:len(testY)-1]
	print(len(testPredict))
	print(len(testY))

	eva_matrics(testY, testPredict)


	plt.plot(testY)
	plt.plot(testPredict)
	plt.show()

	# lstm = load_model('test'+'.h5')
	# testPredict = lstm.predict(trainX)
	# testPredict = scaler.inverse_transform(testPredict)
	# trainY = np.reshape(trainY,(trainY.shape[0],1))
	# trainY = scaler.inverse_transform(trainY)
	# eva_matrics(trainY, testPredict)

	# plt.plot(trainY)
	# plt.plot(testPredict)
	# plt.show()




def use_rfr(trainX,trainY,testX,testY):

	nsamples, nx, ny = trainX.shape
	trainX = trainX.reshape((nsamples,nx*ny))

	nsamples, nx, ny = testX.shape
	testX = testX.reshape((nsamples,nx*ny))

	regr=ensemble.RandomForestRegressor(n_estimators=210, max_features="log2",random_state=50)
	regr.fit(trainX,trainY)
	print("Traing Score:%f"%regr.score(trainX,trainY))
	print("Testing Score:%f"%regr.score(testX,testY))


def use_rfc(trainX,trainY,testX,testY):

	nsamples, nx, ny = trainX.shape
	trainX = trainX.reshape((nsamples,nx*ny))

	nsamples, nx, ny = testX.shape
	testX = testX.reshape((nsamples,nx*ny))

	# 转换为分类问题
	list1 = [-2.0 , -1.7, -1.4 , -1.1, -0.8, -0.5, -0.2, 0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9 ,2.3]
	a = trainY.tolist()
	b = testY.tolist()
	new1 = []
	new2 = []

	for i in range(len(a)):
		y = list(map(lambda x:abs(x-a[i]),list1))
		new1.append(y.index(min(y)))

	for i in range(len(b)):
		m = list(map(lambda x:abs(x-b[i]),list1))
		new2.append(m.index(min(m)))


	new2 = new2[:len(new2)-1]
	new1 = np.array(new1)
	new2 = np.array(new2)


	clf = RandomForestClassifier(n_estimators=830, max_features="log2")

	# class_weight={0: 1, 1: 1, 2:2, 3:3 ,4:4 , 5:5, 6:10, 7:10, 8:10, 9:10, 10:5, 11: 4, 12:3, 13:2 ,14:1}


	clf.fit(trainX,new1)
	print('\nCalculating Accuracy of trained Classifier...')
	acc = clf.score(testX,new2)
	print(acc)

	print('\nMaking Predictions on Validation Data...')
	y_pred = clf.predict(testX)
	print(y_pred)
	print(new2.tolist())

	print('\nCalculating Accuracy of Predictions...')
	accuracy = accuracy_score(new2, y_pred)
	print(accuracy)

	print('\nClassifier Accuracy: ',acc)
	print('\nAccuracy of Classifier on Validation Data: ',accuracy)





if __name__ == '__main__':

	# fix random seed for reproducibility
	numpy.random.seed(7)
	name_list = os.listdir('dataset/.')
	name_list.remove('.DS_Store')
	sz_total = {}
	label = {}
	# 600837.csv

	# dataframe = read_csv('dataset/600837.csv', usecols=[0], engine='python')
	# dataframe = read_csv('dataset/600016.csv',usecols=[3,4,5,6,7,8,9,10,11,12,13], engine='python')
	# 涨跌幅

	label[0] = creat_label()

	Train_set = []
	Train_label = []
	Test_set = []
	Test_label = []

	# create_dataset_(sz_total[0],label[0],4)
	# create_dataset_(sz_total, label, 4)

	for i in range(len(name_list)):
	# for i in range(len(name_list)-40):

		# print(name_list[i])
		dataframe = read_csv('dataset/'+name_list[i], usecols=[7], engine='python')
		# dataframe = read_csv('dataset/601111.csv',usecols=[9], engine='python')
		tmp = dataframe.values.tolist()
		tmp_list = []
		for j in range(len(tmp)):
			tmp_list.append(tmp[j][0])
		tmp_list = numpy.array(tmp_list)
		dataset = numpy.reshape(tmp_list, (tmp_list.shape[0], 1))
		scaler = MinMaxScaler(feature_range=(0, 1))
		dataset = scaler.fit_transform(dataset)

		dataset_spare = dataset
		dataset_test = label[0]

		after_moving_average = moving_average(dataset, 5)
		feature = wavelet(after_moving_average)
		feature = numpy.reshape(feature,(len(feature),1))
		sz_total[i] = feature


		train_size = int(len(feature) * 0.7)
		test_size = len(feature) - train_size

		# print(test_size)
		# print(train_size)


		train, test = feature[0:train_size,:], feature[train_size:len(dataset),:]  #  not the  labels
		train_up, test_up = dataset_test[3:train_size+3,:], dataset_test[train_size+3:len(dataset_test),:]  # 79, 38, are labels 

		# print(len(test))
		# print(len(test_up))

		# print(len(train))
		# print(len(train_up))
		# print("====")


		Train_set.append(train)
		Train_label.append(train_up)
		Test_set.append(test)
		Test_label.append(test_up)

	# print("=======")
	# print(len(Test_set[0]))
	# print(len(Train_set[0]))
	# print("=======")



	# print(Train_set)
	# print(Train_label)
	# print(Test_set)
	# print(Test_label)

	# print(len(Train_set))
	# print(len(Train_label))

	# ===================================== 输入资料建构
	# ===================================== 输入资料建构
	# ===================================== 输入资料建构
	# print(len(Train_set))
	# print(len(Test_set))
	# ===================================== 输入资料建构
	# ===================================== 输入资料建构
	# ===================================== 输入资料建构

	look_back = 4
	length = len(Train_set)
	tmpp = []

	for num1 in range(len(Train_set[0])):
		for num2 in range(len(Train_set)):

			c = Train_set[num2][num1].tolist()[0]
			tmpp.append(c)

	tmppp =[]

	for i in range(len(tmpp)-length*look_back):
		if i%length == 0:
			c = tmpp[i:i+length*look_back]
			tmppp.append(c)

	tmppp = np.array(tmppp)
	tmppp = numpy.reshape(tmppp, (tmppp.shape[0], look_back, length))
	tmppp = np.delete(tmppp, 0, 0)


	# =====================================# =====================================


	length2 = len(Test_set)
	tmpp2 = []

	for num1 in range(len(Test_set[0])):
		for num2 in range(len(Test_set)):
			c = Test_set[num2][num1].tolist()[0]
			tmpp2.append(c)

	tmppp2 =[]

	for i in range(len(tmpp2)-length2*look_back):
		if i%length2 == 0:
			c = tmpp2[i:i+length2*look_back]
			tmppp2.append(c)

	tmppp2 = np.array(tmppp2)
	tmppp2 = numpy.reshape(tmppp2, (tmppp2.shape[0], look_back, length2))
	tmppp2 = np.delete(tmppp2, 0, 0)


	# ===================================== 输出资料建构
	# ===================================== 输出资料建构
	# ===================================== 输出资料建构
	# print(len(Train_label))
	# print(len(Test_label))
	# ===================================== 输出资料建构
	# ===================================== 输出资料建构
	# ===================================== 输出资料建构

	# print(len(Test_label[0]))
	# print(len(Train_label[0]))

	for i in range(len(Train_label)):
		Train_label[i] = np.delete(Train_label[i],-1)
		Train_label[i] = Train_label[i][4:]


	for j in range(len(Test_label)):
		Test_label[j] = Test_label[j][4:]
		Test_label[j] = np.delete(Test_label[j], -1)


	trainX = tmppp
	trainY = Train_label[0]
	testX = tmppp2
	testY = Test_label[0]


	trainX = numpy.vstack((trainX, trainX))
	trainY = trainY.tolist()
	trainY = trainY + trainY
	trainY = np.array(trainY)














	# use_gru(trainX,trainY,testX,testY)
	# use_rfc(trainX,trainY,testX,testY)

