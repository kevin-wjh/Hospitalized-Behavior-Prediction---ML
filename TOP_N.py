import pandas as pd 	

diagnoses = pd.read_csv(r'../data/diagnoses_icd.csv')
diagnoses = diagnoses[diagnoses['icd_version'] == 9]

top_code = diagnoses.groupby(by='icd_code').count()['subject_id'].sort_values(ascending=False)

num = 500 

top_n = top_code[:num]
print(top_n)

diagnoses = diagnoses[diagnoses['icd_code'].isin(top_n.index)]
# diagnoses.to_csv('top_{}.csv'.format(num), index = False)

	