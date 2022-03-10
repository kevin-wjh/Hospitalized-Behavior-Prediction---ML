'''
Yihao:
This is the program which gives prediction mainly using SK-learn as the tool.
Methods applied: One-hot encoding, Desicion tree, SVM & Random-forest classifiers.


Future improvements:
1. Repeated codes in 3 classifiers can be simplified
2. ROC diagrams: how to save?


'''


import pandas as pd
import time


def modelling(sample_size, tt_ratio, balancing, top_N, model_approach):

	# Read data from files
	patients = pd.read_csv('data/patients.csv')
	patients = patients[['subject_id','gender']]

	if top_N == 1:
		diagnoses = pd.read_csv('data/top_500.csv')
	elif top_N == 0:
		diagnoses = pd.read_csv('data/diagnoses_icd.csv')
	else:
		diagnoses = pd.read_csv('data/top_300.csv')
		
	diagnoses = diagnoses[diagnoses['icd_version'] == 9]
	diagnoses = diagnoses[['subject_id','hadm_id','icd_code']]

	# 'keep_default_na' makes sure that 'null' = ''，not 'nan'
	admission = pd.read_excel('data/check.xlsx',keep_default_na = False)
	admission = admission[['subject_id','hadm_id','hour','admission_type','admission_location','hospital_expire_flag']]
	# merging data using groupby, merging duplicate data as one
	test = pd.merge(diagnoses,admission,how = 'left',on = ['subject_id','hadm_id'])
	# join data to get 'gender'
	test_gender = pd.merge(test,patients,how = 'left',on = 'subject_id')
	test_check = test_gender.groupby(['subject_id','hadm_id','hour','admission_type','admission_location','hospital_expire_flag','gender'])['icd_code'].count().reset_index()


	# make data sample according to the size 
	def choose_the_same_scale_data(df, target, N, balancing):

		df_0 = df[df[target] == 0]
		df_1 = df[df[target] == 1]

		# show target 1/0 percentage
		percent = len(df_1) / len(df_0)
		
		# have to make a copy here as 'round' command changes the original variable
		percent_copy = percent
		percent_round = str(round(percent_copy, 3))
		print('Data file 1/0 ratio is {}.'.format(percent_round))


		if balancing == 1:

			check_1 = df_1.sample(n = int(percent * N), random_state = 42, replace = False)
			check_0 = df_0.sample(n = int(N/2), random_state = 42, replace = False)

			repeat_cycle = int(1/percent/2)
			#print('Repeating for {} times.'.format(repeat_cycle))
			check_1 = pd.concat([check_1] * repeat_cycle, ignore_index=True)

		elif balancing == 0:

			check_1 = df_1.sample(n = int(percent * N), random_state = 42, replace = False)
			check_0 = df_0.sample(n = int((1 - percent) * N), random_state = 42, replace = False)

		print('* Positive data = {} VS. Negative data = {} *'.format(len(check_1.index), len(check_0.index)))

		final = check_0.append(check_1)
		final.to_csv('check_1_0.csv', index = False)

		print('* Total sampling size = {} *'.format(len(final)))

		return final

	# sample generating here

	#data = choose_the_same_scale_data(test_check,'hospital_expire_flag',3000)

	data = choose_the_same_scale_data(test_check,'hospital_expire_flag', sample_size, balancing)

	# arrange data

	## Starting modelling
	start_modelling = time.time()

	data = data.drop(['icd_code'], axis=1)
	data = pd.merge(data,diagnoses,how = 'left',on = ['subject_id','hadm_id'])
	print(data)

	data = data[['subject_id','hadm_id','hour','admission_type','admission_location','hospital_expire_flag','gender','icd_code']]

	# manage blank value in 'gender' & 'admission_location'
	for i in range(len(data)):
		if data.loc[i,'gender'] == 'M':
			data.loc[i,'gender'] = 1
		else:
			data.loc[i,'gender'] = 0
		if len(data.loc[i,'admission_location']) == 0:
			data.loc[i,'admission_location'] ='null'

	data['gender'] = data['gender'].astype(float)

	#data.to_excel(r'C:\Users\sudisheng\Desktop\data.xlsx',index = False)
	#data = pd.read_excel(r'C:\Users\sudisheng\Desktop\data.xlsx')

	# manage 'admisson_type', 'admission_location' & 'icd_code', encode to 0/1, and remove/rewrite orginal data
	classify_cols = ['admission_type','admission_location','icd_code']
	check_cols = []

	for i in range(len(classify_cols)):

		col_set = list(set(data[classify_cols[i]]))
		print('%s varibale has %s number of types.'%(classify_cols[i],len(col_set)))

		if classify_cols[i] == 'admission_type' or classify_cols[i] == 'admission_location':
			check_cols.append(col_set)

		for j in range(len(col_set)):
			col_name = col_set[j]
			data[col_name] = 0
			for k in range(len(data)):
				if data.loc[k,classify_cols[i]] == col_set[j]:
					data.loc[k,col_set[j]] = 1

	data = data.drop(classify_cols, axis=1)

	# merge the result data according to sample size, for example 3000
	list_cols = ['subject_id','hadm_id','hour','hospital_expire_flag','gender']
	check_cols.append(list_cols)

	groupby_cols = [i for k in check_cols for i in k]
	final = data.groupby(groupby_cols).sum().reset_index()

	#print(final)
	# export the 'final' data before entering to the modelling, for checking purpose
	final.to_csv('final_df.csv', index = False)
	#print(type(final))


	# deal with the modelling
	import sklearn.model_selection as cross_validation
	final_model = final
	final_model = final_model.drop(['subject_id','hadm_id'], axis=1)
	#final_model.to_csv('final_df.csv', index = False)

	cols = list(final_model.columns)
	cols.remove('hospital_expire_flag')
	X_cols = cols
	target = final_model['hospital_expire_flag']
	data_X = final_model[X_cols]

	test_size = tt_ratio
	train_size = 1 - tt_ratio


	train_data,test_data,train_target,test_target = cross_validation.train_test_split(data_X, target, test_size = test_size, train_size = train_size, random_state = 42)

	# Decision Tree Approach

	if model_approach == 0:

		# build Decision Tree to predict the death (hosipital_expire_flag)
		import sklearn.tree as tree
		# using decision tree
		clf = tree.DecisionTreeClassifier()
		clf.fit(train_data,train_target)

		# give prediction
		train_est = clf.predict(train_data)
		train_est_p = clf.predict_proba(train_data)[:,1]
		test_est = clf.predict(test_data)
		test_est_p = clf.predict_proba(test_data)[:,1]

		## Ending modelling
		end_modelling = time.time()
		modelling_time = end_modelling - start_modelling
		#print('## Modelling time = {} ##'. format(modelling_time))

		# draw ROC
		import sklearn.metrics as metrics
		import matplotlib.pyplot as plt
		# data visulization using seaborn
		import seaborn as sns

		red, blue = sns.color_palette('Set1', 2)
		fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
		fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)

		plt.figure(figsize=[6,6])
		plt.plot(fpr_test, tpr_test, color=blue)
		plt.plot(fpr_train, tpr_train, color=red)
		plt.title('ROC curve')
		plt.show()

		# calculate accuracy
		# larger AUC score, better the accuracy, max score = 1
		#print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

		auc_score_1 = metrics.auc(fpr_test, tpr_test)

		# predict dispatch under 24 hours

		final_data = final
		final_data = final_data.drop(['subject_id','hadm_id'], axis=1)

		final_data['is_hour_24'] = final_data[['hour']].apply(lambda x : 1 if x[0] > 24 else 0, axis = 1)
		target = final_data['is_hour_24']

		cols = list(final_data.columns)
		cols.remove('is_hour_24')
		cols.remove('hour')
		X_cols = cols
		data_X = final_data[X_cols]
		train_data,test_data,train_target,test_target = cross_validation.train_test_split(data_X,target,test_size = 0.3,train_size = 0.7,random_state = 42)

		import sklearn.tree as tree
		# using decision tree
		clf = tree.DecisionTreeClassifier()
		clf.fit(train_data,train_target)

		# give prediction
		train_est = clf.predict(train_data)
		train_est_p=clf.predict_proba(train_data)[:,1]
		test_est=clf.predict(test_data)  
		test_est_p=clf.predict_proba(test_data)[:,1]

		# draw ROC
		import sklearn.metrics as metrics
		import matplotlib.pyplot as plt
		import seaborn as sns

		red,blue = sns.color_palette('Set1',2)
		fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
		fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)
		plt.figure(figsize=[6,6])
		plt.plot(fpr_test, tpr_test, color=blue)
		plt.plot(fpr_train, tpr_train, color=red)
		plt.title('ROC curve')
		plt.show()

		# calculate AUC score
		#print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))
		auc_score_2 = metrics.auc(fpr_test, tpr_test)

		#print(modelling_time, auc_score_1, auc_score_2)


	### SVM Approach
	elif model_approach == 1:

		import matplotlib.pyplot as plt 
		from sklearn import svm, datasets, metrics, model_selection

		from sklearn.metrics import accuracy_score
		clf = svm.SVC(probability=True)
		clf.fit(train_data,train_target)



		## Ending modelling
		end_modelling = time.time()
		modelling_time = end_modelling - start_modelling

		# give prediction
		train_est = clf.predict(train_data)
		train_est_p = clf.predict_proba(train_data)[:,1]
		test_est = clf.predict(test_data)
		test_est_p = clf.predict_proba(test_data)[:,1]


		# draw ROC
		import sklearn.metrics as metrics
		import matplotlib.pyplot as plt
		# data visulization using seaborn
		import seaborn as sns

		red, blue = sns.color_palette('Set1', 2)
		fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
		fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)

		plt.figure(figsize=[6,6])
		plt.plot(fpr_test, tpr_test, color=blue)
		plt.plot(fpr_train, tpr_train, color=red)
		plt.title('ROC curve')
		plt.show()

		# calculate accuracy
		# larger AUC score, better the accuracy, max score = 1
		#print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

		auc_score_1 = metrics.auc(fpr_test, tpr_test)



		final_data = final
		final_data = final_data.drop(['subject_id','hadm_id'], axis=1)

		final_data['is_hour_24'] = final_data[['hour']].apply(lambda x : 1 if x[0] > 24 else 0, axis = 1)
		target = final_data['is_hour_24']

		cols = list(final_data.columns)
		cols.remove('is_hour_24')
		cols.remove('hour')
		X_cols = cols
		data_X = final_data[X_cols]
		train_data,test_data,train_target,test_target = cross_validation.train_test_split(data_X,target,test_size = 0.3,train_size = 0.7,random_state = 42)

		from sklearn import svm
		import matplotlib.pyplot as plt 
		from sklearn import svm, datasets, metrics, model_selection

		from sklearn.metrics import accuracy_score
		clf = svm.SVC(probability=True)
		clf.fit(train_data,train_target)

		# give prediction
		train_est = clf.predict(train_data)
		train_est_p = clf.predict_proba(train_data)[:,1]
		test_est = clf.predict(test_data)
		test_est_p = clf.predict_proba(test_data)[:,1]


		# draw ROC
		import sklearn.metrics as metrics
		import matplotlib.pyplot as plt
		# data visulization using seaborn
		import seaborn as sns

		red, blue = sns.color_palette('Set1', 2)
		fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
		fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)

		plt.figure(figsize=[6,6])
		plt.plot(fpr_test, tpr_test, color=blue)
		plt.plot(fpr_train, tpr_train, color=red)
		plt.title('ROC curve')
		plt.show()

		# calculate accuracy
		# larger AUC score, better the accuracy, max score = 1
		#print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

		auc_score_2 = metrics.auc(fpr_test, tpr_test)


	### Random Forest Approach
	elif model_approach == 2:

		from sklearn.ensemble import RandomForestClassifier
		from sklearn.datasets import make_classification

		clf = RandomForestClassifier(max_depth=2, random_state=0)
		clf.fit(train_data,train_target)

		# give prediction
		train_est = clf.predict(train_data)
		train_est_p = clf.predict_proba(train_data)[:,1]
		test_est = clf.predict(test_data)
		test_est_p = clf.predict_proba(test_data)[:,1]

		## Ending modelling
		end_modelling = time.time()
		modelling_time = end_modelling - start_modelling


		# draw ROC
		import sklearn.metrics as metrics
		import matplotlib.pyplot as plt
		# data visulization using seaborn
		import seaborn as sns

		red, blue = sns.color_palette('Set1', 2)
		fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
		fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)

		plt.figure(figsize=[6,6])
		plt.plot(fpr_test, tpr_test, color=blue)
		plt.plot(fpr_train, tpr_train, color=red)
		plt.title('ROC curve')
		plt.show()

		# calculate accuracy
		# larger AUC score, better the accuracy, max score = 1
		#print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

		auc_score_1 = metrics.auc(fpr_test, tpr_test)


		final_data = final
		final_data = final_data.drop(['subject_id','hadm_id'], axis=1)

		final_data['is_hour_24'] = final_data[['hour']].apply(lambda x : 1 if x[0] > 24 else 0, axis = 1)
		target = final_data['is_hour_24']

		cols = list(final_data.columns)
		cols.remove('is_hour_24')
		cols.remove('hour')
		X_cols = cols
		data_X = final_data[X_cols]
		train_data,test_data,train_target,test_target = cross_validation.train_test_split(data_X,target,test_size = 0.3,train_size = 0.7,random_state = 42)

		from sklearn.ensemble import RandomForestClassifier
		from sklearn.datasets import make_classification

		clf = RandomForestClassifier(max_depth=2, random_state=0)
		clf.fit(train_data,train_target)

		# give prediction
		train_est = clf.predict(train_data)
		train_est_p = clf.predict_proba(train_data)[:,1]
		test_est = clf.predict(test_data)
		test_est_p = clf.predict_proba(test_data)[:,1]


		# draw ROC
		import sklearn.metrics as metrics
		import matplotlib.pyplot as plt
		# data visulization using seaborn
		import seaborn as sns

		red, blue = sns.color_palette('Set1', 2)
		fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
		fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)

		plt.figure(figsize=[6,6])
		plt.plot(fpr_test, tpr_test, color=blue)
		plt.plot(fpr_train, tpr_train, color=red)
		plt.title('ROC curve')
		plt.show()

		# calculate accuracy
		# larger AUC score, better the accuracy, max score = 1
		#print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

		auc_score_2 = metrics.auc(fpr_test, tpr_test)


	print('Modelling Time is {} ms.'.format(round(modelling_time, 4)))
	print('AUC Score 1 is {}.'.format(round(auc_score_1, 4)))
	print('AUC Score 2 is {}.'.format(round(auc_score_2, 4)))

	return float(modelling_time), float(auc_score_1), float(auc_score_2)



if __name__ == '__main__':

	#modelling(sample_size, tt_ratio, balancing, top_N, model_approach)

	modelling(300, 0.3, 1, 1, 2)
