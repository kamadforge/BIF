# this script is copied from
# https://www.kaggle.com/victoralcimed/diabetes-readmission-through-logistic-regression/notebook
# we follow the undersampling

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
# import pandas_profiling
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, precision_score, recall_score
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
# from imblearn.over_sampling import SMOTE

### data loading ###
bdd = pd.read_csv('diabetic_data.csv')

bdd = bdd.sort_values("encounter_id").reset_index(drop=True)
bdd.head()

bdd["uno"] = 1
bdd["Objective"] = False
bdd.loc[bdd.readmitted == "<30", "Objective"] = True
bdd = bdd.drop(["readmitted"], axis = 1)

sns.countplot(bdd["Objective"])
# plt.show()

bdd["alreadyCame"] = True
bdd.loc[bdd.duplicated("patient_nbr", keep="first"), "alreadyCame"] = False
bdd.loc[bdd.duplicated("patient_nbr", keep=False),:].sort_values(["patient_nbr", "encounter_id"]).head(20)

bdd.loc[bdd.change == "Ch", "change"] = True
bdd.loc[bdd.change == "No", "change"] = False

bdd.loc[bdd.diabetesMed == "Yes", "diabetesMed"] = True
bdd.loc[bdd.diabetesMed == "No", "diabetesMed"] = False

fig, axs = plt.subplots(1, 2, figsize = (10, 5))
sns.countplot(data = bdd, x ="change", hue = "Objective", ax = axs[0])
sns.countplot(data = bdd, x ="diabetesMed", hue = "Objective", ax = axs[1])
# plt.show()

bdd["A1C"] = bdd["A1Cresult"]
bdd.loc[bdd.A1Cresult.isin([">7", ">8"]), "A1C"] = "Abnorm"
fig, axs = plt.subplots(1, 2, figsize = (10, 5))
sns.countplot(data = bdd, x = "A1Cresult", hue = "Objective", ax = axs[0])
sns.countplot(data = bdd, x = "A1C", hue = "Objective", ax = axs[1])
# plt.show()

bdd = pd.get_dummies(data = bdd, columns = ["A1C"], prefix = "A1C", drop_first=False)
bdd = bdd.drop(["A1Cresult", "A1C_None"], axis = 1)

bdd["GluSerum"] = bdd["max_glu_serum"]
bdd.loc[bdd.max_glu_serum.isin([">200", ">300"]), "GluSerum"] = "Abnorm"
fig, axs = plt.subplots(1, 2, figsize = (10, 5))
sns.countplot(data = bdd, x = "max_glu_serum", hue = "Objective", ax = axs[0])
sns.countplot(data = bdd, x = "GluSerum", hue = "Objective", ax = axs[1])
# plt.show()

bdd = pd.get_dummies(data = bdd, columns = ["GluSerum"], prefix = "GluSerum", drop_first=False)
bdd = bdd.drop(["max_glu_serum", "GluSerum_None"], axis = 1)

sns.countplot(data = bdd, x = "gender", hue = "Objective")
# plt.show()

bdd["isFemale"] = False
bdd.loc[bdd.gender == "Female", "isFemale"] = True
bdd = bdd[bdd.gender != "Unknown/Invalid"]

people_multiple_gender = bdd.loc[(bdd.duplicated("patient_nbr", keep=False)), ["patient_nbr", "gender", "uno"]].groupby(
    ["patient_nbr", "gender"]).count().reset_index()
people_multiple_gender = people_multiple_gender[people_multiple_gender.duplicated("patient_nbr", keep=False)]

list_nb_to_drop = []

for nb in people_multiple_gender.patient_nbr.unique():
    nbmin = 1
    suppr = False
    value = ""

    for sex in people_multiple_gender.loc[people_multiple_gender.patient_nbr == nb, "gender"]:
        if people_multiple_gender.loc[
            (people_multiple_gender.patient_nbr == nb) & (people_multiple_gender.gender == sex), "uno"].values[
            0] == nbmin:
            suppr = True
        else:
            suppr = False
            nbmin = people_multiple_gender.loc[
                (people_multiple_gender.patient_nbr == nb) & (people_multiple_gender.gender == sex), "uno"]
            value = sex

    if suppr:
        list_nb_to_drop.append(nb)
    else:
        bdd.loc[(bdd.patient_nbr == nb), "gender"] = value

bdd = bdd[~bdd.patient_nbr.isin(list_nb_to_drop)]
bdd = bdd.drop("gender", axis=1)

sns.countplot(data = bdd, x = "race", hue = "Objective")
# plt.show()

people_multiple_race = bdd.loc[(bdd.duplicated("patient_nbr", keep=False)), ["patient_nbr", "race", "uno"]].groupby(
    ["patient_nbr", "race"]).count().reset_index()
people_multiple_race = people_multiple_race[people_multiple_race.duplicated("patient_nbr", keep=False)]

list_nb_to_drop = []

for nb in people_multiple_race.patient_nbr.unique():

    list_race = list(people_multiple_race.loc[people_multiple_race.patient_nbr == nb, "race"].unique())
    try:
        list_race.remove("?")
    except:
        "Nothing"

    if len(list_race) == 1:
        # print(list_race[0])
        bdd.loc[(bdd.patient_nbr == nb), "race"] = list_race[0]
    else:
        nbmin = 1
        suppr = False
        value = ""

        for rac in list_race:
            if people_multiple_race.loc[
                (people_multiple_race.patient_nbr == nb) & (people_multiple_race.race == rac), "uno"].values[
                0] == nbmin:
                suppr = True
            else:
                suppr = False
                nbmin = people_multiple_race.loc[
                    (people_multiple_race.patient_nbr == nb) & (people_multiple_race.race == rac), "uno"].values[0]
                value = rac

    if suppr:
        list_nb_to_drop.append(nb)
    else:
        bdd.loc[(bdd.patient_nbr == nb), "race"] = value

bdd = bdd[~bdd.patient_nbr.isin(list_nb_to_drop)]

bdd.loc[bdd.race == "?", "race"] = "unavailable"

bdd = pd.get_dummies(data = bdd, columns = ["race"], prefix = "race", drop_first=False)
bdd = bdd.drop("race_unavailable", axis = 1)


def norm_diag(bdd, diag):
    if bdd[diag] == "?":
        return "Unavailable"
    elif bdd[diag][0] == "E":
        return "Other"
    elif bdd[diag][0] == "V":
        return "Other"
    else:
        num = float(bdd[diag])

        if np.trunc(num) == 250:
            return "Diabetes"
        elif num <= 139:
            return "Other"
        elif num <= 279:
            return "Neoplasms"
        elif num <= 389:
            return "Other"
        elif num <= 459:
            return "Circulatory"
        elif num <= 519:
            return "Respiratory"
        elif num <= 579:
            return "Digestive"
        elif num <= 629:
            return "Genitourinary"
        elif num <= 679:
            return "Other"
        elif num <= 709:
            return "Neoplasms"
        elif num <= 739:
            return "Musculoskeletal"
        elif num <= 759:
            return "Other"
        elif num in [780, 781, 782, 783, 784]:
            return "Neoplasms"
        elif num == 785:
            return "Circulatory"
        elif num == 786:
            return "Respiratory"
        elif num == 787:
            return "Digestive"
        elif num == 788:
            return "Genitourinary"
        elif num == 789:
            return "Digestive"
        elif num in np.arange(790, 800):
            return "Neoplasms"
        elif num >= 800:
            return "Injury"
        else:
            return num

bdd["diag_1_norm"] = bdd.apply(norm_diag, axis=1, diag="diag_1")
bdd["diag_2_norm"] = bdd.apply(norm_diag, axis=1, diag="diag_2")
bdd["diag_3_norm"] = bdd.apply(norm_diag, axis=1, diag="diag_3")

list_diag = ['Circulatory', 'Neoplasms', 'Diabetes', 'Respiratory', 'Other', 'Injury', 'Musculoskeletal', 'Digestive', 'Genitourinary']

fig, axs = plt.subplots(3, 1, figsize = (15, 10))
sns.countplot(data = bdd, y = "diag_1_norm", hue = "Objective", ax = axs[0], order = list_diag)
sns.countplot(data = bdd, y = "diag_2_norm", hue = "Objective", ax = axs[1], order = list_diag)
sns.countplot(data = bdd, y = "diag_3_norm", hue = "Objective", ax = axs[2], order = list_diag)
# plt.show()

def diag_atleast (bdd, val) :
    if (bdd["diag_1_norm"] == val) | (bdd["diag_2_norm"] == val) | (bdd["diag_3_norm"] == val) :
        return True
    else :
        return False

for val in list_diag :
    name_var = "diag_atleast_"+ val
    print(name_var)
    bdd[name_var] = bdd.apply(diag_atleast, axis = 1, val=val)

list_diag_inter = list_diag.copy()

for diag in list_diag:
    list_diag_inter.remove(diag)

    for diag2 in list_diag_inter:
        name = "diag_" + diag + "_&_" + diag2
        bdd[name] = (bdd["diag_atleast_" + diag] & bdd["diag_atleast_" + diag2])


bdd = bdd.drop(["diag_1", "diag_2", "diag_3"], axis = 1)


medoc = ['metformin', 'repaglinide', 'nateglinide',
       'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
       'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
       'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
       'insulin', 'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone']

medoc_rare = ["nateglinide", "chlorpropamide", "acetohexamide", "tolbutamide",
             "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
             "citoglipton", "glyburide-metformin", "glipizide-metformin",
             "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]

medoc_usuels = [med for med in medoc if med not in medoc_rare]

for med in medoc_usuels:
    name = "take_" + med
    bdd[name] = bdd[med].isin(["Down", "Steady", "Up"])

medoc_inter = medoc_usuels.copy()

for med in medoc_usuels:

    medoc_inter.remove(med)

    for med2 in medoc_inter:
        name = "take_" + med + "_&_" + med2
        bdd[name] = (bdd["take_" + med] & bdd["take_" + med2])


def nbMedocRare (bdd, listMedoc) :
    nb = 0
    for med in listMedoc :
        if bdd[med] != "No" :
            nb += 1
    return nb

bdd["nb_rare_medoc"] = bdd.apply(nbMedocRare, listMedoc = medoc_rare, axis = 1)

bdd['admission_type_id'] = bdd['admission_type_id'].replace([1, 2, 7], "emergency")
bdd['admission_type_id'] = bdd['admission_type_id'].replace([4, 5, 6, 8], "unavailable")
bdd['admission_type_id'] = bdd['admission_type_id'].replace(3, "elective")

bdd = pd.get_dummies(data = bdd, columns = ["admission_type_id"], prefix="admission_type", drop_first=False)
bdd = bdd.drop(["admission_type_unavailable"], axis = 1)

bdd['admission_source_id'] = bdd['admission_source_id'].replace([1, 2, 3], "referral")
bdd['admission_source_id'] = bdd['admission_source_id'].replace([4, 5, 6, 10, 22, 25], "transfert")
bdd['admission_source_id'] = bdd['admission_source_id'].replace([8, 14, 11, 13, 9, 15, 17, 20, 21], "unavailable")
bdd['admission_source_id'] = bdd['admission_source_id'].replace(7, "emergencyRoom")

bdd = pd.get_dummies(data = bdd, columns = ["admission_source_id"], prefix="admission_source", drop_first=False)
bdd = bdd.drop(["admission_source_unavailable"], axis = 1)


bdd['discharge_disposition_id'] = bdd['discharge_disposition_id'].replace([1, 6, 8, 9, 10], "home")
bdd['discharge_disposition_id'] = bdd['discharge_disposition_id'].replace([2, 3, 4, 5, 14, 22, 23, 24], "transfert")
bdd['discharge_disposition_id'] = bdd['discharge_disposition_id'].replace([18, 25, 26], "unavailable")
bdd['discharge_disposition_id'] = bdd['discharge_disposition_id'].replace([7, 10, 11, 13, 12, 15, 16, 17, 19, 20, 27, 28], "other")

bdd = pd.get_dummies(data = bdd, columns = ["discharge_disposition_id"], prefix="discharge_type", drop_first=False)
bdd = bdd.drop(["discharge_type_unavailable"], axis = 1)


var_quanti = ["time_in_hospital", 'num_lab_procedures', 'num_procedures',
              'num_medications', 'number_outpatient', 'number_emergency',
              'number_inpatient','number_diagnoses', 'nb_rare_medoc']


def recup_age (bdd) :
    return int(bdd.age[-4::].replace('-', '').replace(')', ''))

bdd["age_num"] = bdd.apply(recup_age, axis = 1)
var_quanti.append("age_num")
bdd = bdd.drop("age", axis = 1)


def count_num_medoc(bdd) :
    nb = 0
    for med in medoc :
        if bdd[med] != "No" :
            nb += 1
    return nb

def count_num_medoc_chgmnt(bdd) :
    nb = 0
    for med in medoc :
        if (bdd[med] != "No") & (bdd[med] != "Steady") :
            nb += 1
    return nb

bdd["num_medo_arrived"] = bdd.apply(count_num_medoc, axis = 1)
bdd["num_medo_chgmnt"] = bdd.apply(count_num_medoc_chgmnt, axis = 1)

var_quanti.append("num_medo_arrived")
var_quanti.append("num_medo_chgmnt")


bdd["proportion_chgmnt"] = bdd["num_medo_chgmnt"] / bdd["num_medo_arrived"]
bdd["proportion_chgmnt"] = bdd["proportion_chgmnt"].fillna(0)
var_quanti.append("proportion_chgmnt")


bdd["number_outpatient"] = np.log1p(bdd["number_outpatient"])
bdd["number_emergency"] = np.log1p(bdd["number_emergency"])


bdd = bdd.drop(["weight", "medical_specialty", 'encounter_id', 'patient_nbr', "payer_code", 'diag_1_norm', 'diag_2_norm', 'diag_3_norm'] + medoc, axis = 1)
bdd = bdd.drop("uno", axis = 1)

X = bdd.drop(["Objective"], axis=1)
y = bdd["Objective"]
print('Original dataset shape {}'.format(Counter(y)))

X.insert(0, "Intercept", 1)


undersample = RandomUnderSampler(random_state=42)
new_X, new_y = undersample.fit_resample(X, y)

print('undersampled dataset shape {}'.format(Counter(new_y)))


def reduction_variable_logit(X_train, y_train, showVarToDel=False):
    ultime_model = False
    var_to_del = []

    while (ultime_model == False):
        log_reg = sm.Logit(y_train, X_train.drop(var_to_del, axis=1).astype(float)).fit(maxiter=100, disp=False)

        max_pvalue = max(log_reg.pvalues)

        if max_pvalue < 0.05:
            ultime_model = True
        else:
            varToDel = log_reg.pvalues.index[log_reg.pvalues == max(log_reg.pvalues)].values[0]
            if showVarToDel:
                print(varToDel + ", p-value = " + str(max(log_reg.pvalues)))
            var_to_del.append(varToDel)

    return log_reg, var_to_del

# remove some input features based on its significance
log_reg, var_to_del = reduction_variable_logit(new_X, new_y, showVarToDel=True)
new_X = new_X.drop(var_to_del, axis=1)
X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=0.2, random_state=42)

# test logistic regression to make sure the pre-processing is done correctly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
model = LogisticRegression(solver='lbfgs', max_iter=50000, tol=1e-12)
model.fit(X_train, y_train)
pred = model.predict(X_test)
roc = roc_auc_score(y_test, pred)
prc = average_precision_score(y_test, pred)
print('roc is %f and prc %f' %(roc, prc))

# store all the variables so we can load them in our test script
X_tr = X_train.values.astype(np.float64)
X_tst = X_test.values.astype(np.float64)
y_tr = y_train.values.astype(np.float64)
y_tst = y_test.values.astype(np.float64)

column_names = X_train.columns.values.tolist()


np.save('X_tr_diabetic.npy', X_tr)
np.save('X_tst_diabetic.npy', X_tst)
np.save('y_tr_diabetic.npy', y_tr)
np.save('y_tst_diabetic.npy', y_tst)

np.save('column_names_diabetic.npy', column_names)