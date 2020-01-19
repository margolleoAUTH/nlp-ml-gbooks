import requests
import matplotlib.pyplot as plt
import seaborn as sns
import math
import collections
import numpy as np
import datetime
from myPreProcessor import MyPreProcessor
from myDBHandler import MyDBHandler
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SVMSMOTE
from imblearn.metrics import geometric_mean_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import TruncatedSVD
from imblearn.under_sampling import RandomUnderSampler

# ===================================================================================================
# IMPORTANT NOTE!!!
# Unfortunately, our IDE had already installed some libraries before we started the implementation.
# This was a trap for us and as a result we did not maintain the requirement.txt file.
# So sorry! requirements.txt is DEPRECATED do not run pip install using this configuration file
# ===================================================================================================

# Extra Comments
#
# git config --replace-all user.name "margolleoAUTH"
# git config --replace-all user.email "margolleo@csd.auth.com"
# git commit --author="margolleoAUTH <margolleo@csd.auth.com>"
#
# Database(Atlas):
# https://cloud.mongodb.com/user?n=%2Fv2%2F5ce5e9a1ff7a252fdb34b8b1&nextHash=%23clusters#/atlas/login
# margolleo@csd.auth.com
# 5Kleanthis@
# Please add your IP to the whitelist in order to connect with the database(We do not have open firewall - 0.0.0.0)
# preprocessor lib does not work for pip install - needs to be installed manually


def distinct_features_labels(_allX, _allCategories):
    print("___________________________________________________________________________________________________")
    _XTemp = []
    _yTemp = []
    _i = 0
    print("Number of Features/Categories - Selected:: " + str(len(_allX)))
    for _index, _item in enumerate(_allX):
        if _item not in _XTemp:
            _XTemp.append(_item)
            _yTemp.append(_allCategories[_index])
        else:
            _i += 1
    print("Number of Features/Categories - Duplicated:: " + str(_i))
    return _XTemp, _yTemp


def data_manipulation(_list, _joinString):
    _listManipulated = []
    for _item in _list:
        _item = preProcessor.data_pre_processing_nltk_extended(_item)
        _toSpaceString = _joinString.join(_item)
        _listManipulated.append(_toSpaceString)
    return _listManipulated


def data_printer(_list, _listName, _message):
    print("___________________________________________________________________________________________________")
    print(_listName + " - Data printing: ")
    _counter = collections.Counter(_list)
    print(_counter)
    print(_message + str(len(_counter)))


def select_n_components(_var_ratio, _goal_var):
    _total_variance = 0.0
    _n_components = 0
    for explained_variance in _var_ratio:
        _total_variance += explained_variance
        _n_components += 1
        if _total_variance >= _goal_var:
            break
    return _n_components


def data_confusion_matrix(_modeName, _y_test, _y_predicted):
    _y_test_set = sorted(set(_y_test))
    _y_test_size = len(_y_test_set)
    _y_predicted_set = sorted(set(_y_predicted))
    _y_predicted_size = len(_y_predicted_set)

    _CM = confusion_matrix(_y_test, _y_predicted)

    if _y_test_size > _y_predicted_size:
        sns.heatmap(_CM, cmap="Reds", linewidths=0.1, annot=True, cbar=False,
                    xticklabels=_y_test_set, yticklabels=_y_test_set, annot_kws={"size": 8},
                    fmt="d")
    else:
        sns.heatmap(_CM, cmap="Reds", linewidths=0.1, annot=True, cbar=False,
                    xticklabels=_y_predicted_set, yticklabels=_y_predicted_set, annot_kws={"size": 8},
                    fmt="d")

    _bottom, _top = plt.ylim()
    _bottom += 0.5
    _top -= 0.5
    plt.ylim(_bottom, _top)
    plt.title(_modeName + "Confusion Matrix")
    plt.xlabel("Predicted output")
    plt.ylabel("True output")
    # plt.show()
    with PdfPages(_modeName + "HeatMap.pdf") as export_pdf:
        export_pdf.savefig()
    plt.close()

    _FP = 0
    _FN = 0
    _TP = 0
    _TN = 0
    _PRECISION = 0
    _cm_length = int(math.sqrt(_CM.size))
    for _i in range(_cm_length):
        _tFP = 0
        _tFN = 0
        _tTP = 0
        _tTN = 0
        for _j in range(_cm_length):
            if _j == _i:
                _TP += _CM[_i][_j]
                _tTP += _CM[_i][_j]
            else:
                _FN += _CM[_i][_j]
                _tFN += _CM[_i][_j]
                _FP += _CM[_j][_i]
                _tFP += _CM[_j][_i]
        if 0 == (_tTP + _tFP):
            _PRECISION = 0
        else:
            _PRECISION += _tTP / (_tTP + _tFP)

    # print("My micro Precision: ", round(_TP * 100 / (_TP + _FP), 5))
    # print("My macro Precision: ", round(_PRECISION * 100 / _cm_length, 5))


def sample_handler(_x_train, _y_train):
    # _nm1 = RandomUnderSampler(replacement=True)
    # _x_train, _y_train = _nm1.fit_resample(_x_train, _y_train)

    _sm = SVMSMOTE()
    _x_train, _y_train = _sm.fit_resample(_x_train, _y_train)

    return _x_train, _y_train


def result_handler(_model, _x_train, _x_test, _y_train, _y_test, _is_grid_search, _modeName=None):
    _model.fit(_x_train, _y_train)
    _y_predicted = _model.predict(_x_test)

    _accuracy = round(accuracy_score(_y_test, _y_predicted) * 100, 5)
    _recall = round(recall_score(_y_test, _y_predicted, average="macro") * 100, 5)
    _precision_macro = round(precision_score(_y_test, _y_predicted, average="macro") * 100, 5)
    _f1 = round(f1_score(_y_test, _y_predicted, average="macro") * 100, 5)
    _gMean = round(geometric_mean_score(_y_test, _y_predicted, average="macro")* 100, 5)
    if _is_grid_search:
        data_confusion_matrix(_modeName, _y_test, _y_predicted)
        return _accuracy, _recall, _precision_macro, _f1, _gMean, _model.best_estimator_
    else:
        return _accuracy, _recall, _precision_macro, _f1, _gMean


def result_printer(_accuracy, _recall, _precision, _f1, _gMean, _modelName, _modelFirst, _modelSecond, _folds):
    print("___________________________________________________________________________________________________")
    if _modelSecond is None:
        if "MultinomialNB " == _modelName:
            print(_modelFirst)
        else:
            print(_modelFirst.get_params())
    else:
        print(_modelFirst.get_params())
        print(_modelSecond.get_params())

    print(_modelName + "Result printing: ")
    print("_______________________________")
    print(_modelName + "Accuracy: ", round(_accuracy/_folds, 5))
    print(_modelName + "Recall: ", round(_recall/_folds, 5))
    print(_modelName + "Precision: ", round(_precision/_folds, 5))
    print(_modelName + "F1: ", round(_f1/_folds, 5))
    print(_modelName + "G-Mean: ", round(_gMean/_folds, 5))


def le_cosine(_X, _y, _yFactor):
    _cosine_distance = 1 - cosine_similarity(_X)
    _json = {}
    for _i_index, _item in enumerate(_cosine_distance):
        _json[str(_i_index) + "FP"] = ""
        _json[str(_i_index) + "FN"] = ""
        for _j_index, _j in enumerate(_item):
            if _j < 0.66:
                _json[str(_i_index) + "FP"] += str(_y[_j_index]) + ","
            if _j > 0.99:
                _json[str(_i_index) + "FN"] += str(_y[_j_index]) + ","

    _exclude = []
    _counter_able = ""
    _fp = 0
    _pair_index = 0
    for _index, _item in enumerate(_json):
        if _index % 2 == 0:
            _pair_index = _index
            _fp = _json[_item].split(",")
            _counter_able = _fp[0]
            _fp = len(set(_fp))
        else:
            if _fp > _yFactor:
                _exclude.append(int(_pair_index/2))

    _XCosine = np.delete(_X.toarray(), _exclude, 0)
    _yCosine = np.delete(_y, _exclude, 0)
    print("Number of Cosine Similarity Features - Excluded:: " + str(len(_exclude)))
    return sparse.csr_matrix(_XCosine), np.array(_yCosine)


if __name__ == "__main__":

    try:
        print("_______________________________")
        print(datetime.datetime.now().strftime("%D %H:%M:%S"))
        # FETCH4STORE = True
        # ML = False
        FETCH4STORE = False
        ML = True
        QUERY = ""

        DB = "gbooks"
        DATA_COL = "data_" + QUERY
        CAT_COL = "categories_" + QUERY

        # Initialization of Controller's Instances
        # Database Handler
        # Pre-Processor
        preProcessor = MyPreProcessor()
        databaseHandler = MyDBHandler(DB)

        if FETCH4STORE:
            print("Fetching data for db-storing started ...")
            databaseHandler.delete_all_data(DATA_COL)
            databaseHandler.delete_all_data(CAT_COL)

            # API-Endpoint
            URL = "https://www.googleapis.com/books/v1/volumes"
            # PARAMS = {"q": QUERY, "key": "AIzaSyD3etuN-ZjEVM85annR8Gc0sb1Z3phWCqA"}
            # PARAMS = {"q": QUERY, "key": "AIzaSyA3T_IBqnf4zkWjlEjMj4NomvoCf-P-nFE"}
            # PARAMS = {"q": QUERY, "key": "AIzaSyCdZPzYO3cZg89P7ATWPxcQSNYpkRmNCls"}
            # PARAMS = {"q": QUERY, "key": "AIzaSyBo90b_VuOYcgbiD9hFIWXdDLwweSwtQLA"}
            # PARAMS = {"q": QUERY, "key": "AIzaSyCsv_2gdLYhpKnz3CdGDykB4HZB-ZZKIsE"}
            # PARAMS = {"q": QUERY, "key": "AIzaSyCc-2TofUelwwh5sRNW4Uz-EgWStfM_jqw"}
            # PARAMS = {"q": QUERY, "key": "AIzaSyBT7Y4p9r8PIWqmoP4sJnULGGUJCafUesY"}
            # PARAMS = {"q": QUERY, "key": "AIzaSyD_daqJ7ToX5BIcaLa5EL71PsM8hAqrxr8"}
            # PARAMS = {"q": QUERY, "key": "AIzaSyDcgDVEHAAETDM4cArfA_QIGfqu4BWPHtg"}
            PARAMS = {"q": QUERY, "key": "AIzaSyBUTnJnda81h7AJ3pmFUB519RtDigzHulo"}

            X = []
            next_page = True
            start_index = 0
            while next_page:
                PARAMS["startIndex"] = start_index
                request_data = requests.get(url=URL, params=PARAMS).json()
                if "items" in request_data:
                    data = request_data["items"]
                    for item in data:
                        if "title" in item["volumeInfo"] and "description" in item["volumeInfo"] and "categories" in \
                                item["volumeInfo"]:
                            item_data = item["volumeInfo"]["title"]
                            item_data += " " + item["volumeInfo"]["description"]
                            item_category = item["volumeInfo"]["categories"][0]
                            if item_data not in X:
                                X.append(item_data)
                                databaseHandler.insert_data(DATA_COL, {"item": item_data})
                                databaseHandler.insert_data(CAT_COL, {"item": item_category})
                else:
                    if "error" in request_data and request_data["error"]["code"] == 403:
                        raise Exception(request_data)
                    else:
                        next_page = False
                start_index += 1
            print("Fetching data for db-storing performed successfully!")

        if ML:
            print("Machine Learning started ...")
            all = databaseHandler.get_all_data([
                "data_abnormal",
                "data_abortion",
                "data_absorption",
                "data_acute",
                "data_adaptation",
                "data_agent",
                "data_allele",
                "data_allergen",
                "data_allergy",
                "data_amniocentesis",
                "data_anthrax",
                "data_antibiotic",
                "data_antibodies",
                "data_antibody",
                "data_antidote",
                "data_antigen",
                "data_antioxidant",
                "data_antiviral",
                "data_arteriosclerosis",
                "data_ash",
                "data_assembler",
                "data_atom",
                "data_atropine",
                "data_autosome",
                "data_bacillus",
                "data_bacteria",
                "data_bark",
                "data_base",
                "data_bioavailability",
                "data_biochemistry",
                "data_biodiesel",
                "data_biodiversity",
                "data_bioeconomics",
                "data_bioelectrical",
                "data_bioenergy",
                "data_bioengineering",
                "data_bioethics",
                "data_biofuels",
                "data_biogas",
                "data_biogeography",
                "data_bioimpedance",
                "data_bioinformatics",
                "data_biology",
                "data_biomarker",
                "data_biomass",
                "data_biomaterials",
                "data_biome",
                "data_biomedical",
                "data_biomedicine",
                "data_biopharming",
                "data_biophysics",
                "data_biopower",
                "data_biorefinery",
                "data_bioregion",
                "data_bioremediation",
                "data_biostatistics",
                "data_biota",
                "data_biotechnology",
                "data_bioterrorism",
                "data_blastocyst",
                "data_body",
                "data_botanical",
                "data_botany",
                "data_botulism",
                "data_brucellosis",
                "data_buckminsterfullerene",
                "data_calorie",
                "data_carbohydrate",
                "data_carcinogen",
                "data_carotenoids",
                "data_carrier",
                "data_catalyst",
                "data_cell",
                "data_cellulose",
                "data_chelation",
                "data_chips",
                "data_chorion",
                "data_chromatography",
                "data_chromosome",
                "data_class",
                "data_clone",
                "data_cloning",
                "data_clostridium",
                "data_communicable",
                "data_community",
                "data_composites",
                "data_congenital",
                "data_consanguinity",
                "data_contraindication",
                "data_cytogenetic"
            ], "item")
            allX = all[0]
            allCategories = all[1]

            XTemp, yTemp = distinct_features_labels(allX, allCategories)

            XTempManipulated = data_manipulation(XTemp, " ")
            yTempManipulated = data_manipulation(yTemp, "")

            print("Number of Features - Included:: ", len(XTemp))
            data_printer(yTempManipulated, "yTempManipulated", "Number of Categories - Included:: ")
            counter = collections.Counter(yTempManipulated)

            X = []
            y = []
            i = 0
            for index, item in enumerate(yTempManipulated):
                yCount = counter[item]
                yFactor = yCount/len(yTempManipulated)
                if yFactor > 0.03:
                    y.append(yTempManipulated[index])
                    X.append(XTempManipulated[index])
                    i += 1
            print("Number of Filtered Features - Included:: " + str(i))
            data_printer(y, "y", "Number of Filtered Categories - Included:: ")

            labelEncoder = LabelEncoder()
            y = labelEncoder.fit_transform(y)

            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(X)

            # _tsvd = _TruncatedSVD()
            # _tsvd.fit(X)
            # _n_components = select_n_components(_tsvd.explained_variance_ratio_, 0.95)
            # print("TruncatedSVD space:: " + str(_n_components))
            # _truncatedSVD = TruncatedSVD(n_components=_n_components)

            X, y = le_cosine(X, y, 2)

            accuracyMultinomialNB = 0
            recallMultinomialNB = 0
            precisionMacroMultinomialNB = 0
            f1MultinomialNB = 0
            gMeanMultinomialNB = 0
            alpha = 0.1
            folds = 5

            kf = StratifiedKFold(n_splits=folds)
            for train_index, test_index in kf.split(X, y):
                x_train = X[train_index]
                y_train = y[train_index]
                x_test = X[test_index]
                y_test = y[test_index]

                x_train, y_train = sample_handler(x_train, y_train)
                minMaxScaler = MinMaxScaler()
                x_train_trans = minMaxScaler.fit_transform(x_train.todense())
                x_test_trans = minMaxScaler.transform(x_test.todense())
                x_train = sparse.csr_matrix(x_train_trans)
                x_test = sparse.csr_matrix(x_test_trans)

                multinomialNB = MultinomialNB(alpha=alpha)
                resultsMultinomialNB = result_handler(multinomialNB, x_train, x_test, y_train, y_test, False)

                accuracyMultinomialNB += resultsMultinomialNB[0]
                recallMultinomialNB += resultsMultinomialNB[1]
                precisionMacroMultinomialNB += resultsMultinomialNB[2]
                f1MultinomialNB += resultsMultinomialNB[3]
                gMeanMultinomialNB += resultsMultinomialNB[4]

            x_train, x_test, y_train, y_test = train_test_split(X, y)
            x_train, y_train = sample_handler(x_train, y_train)

            paramsLogisticRegression = {"solver": ("sag", "lbfgs")}
            paramsSVC = {"kernel": ("linear", "rbf"), "C": [0.1, 1, 10]}
            logisticRegression = LogisticRegression(multi_class="multinomial", class_weight="balanced", max_iter=100)
            SVC = SVC(gamma="auto", class_weight="balanced")
            gridLogisticRegression = GridSearchCV(logisticRegression, paramsLogisticRegression, cv=folds)
            gridSVC = GridSearchCV(SVC, paramsSVC, cv=folds)

            resultsLogisticRegression = result_handler(gridLogisticRegression, x_train, x_test, y_train, y_test, True, "LogisticRegression ")
            resultsSVC = result_handler(gridSVC, x_train, x_test, y_train, y_test, True, "SVC ")

            estimators = [
                ("logisticRegression", gridLogisticRegression),
                ("svm", resultsSVC[5])
            ]
            stackingLogisticRegression = StackingClassifier(estimators=estimators, final_estimator=gridLogisticRegression)
            stackingSVC = StackingClassifier(estimators=estimators, final_estimator=resultsSVC[5])

            resultsStackingLogisticRegression = result_handler(stackingLogisticRegression, x_train, x_test, y_train, y_test, False)
            resultsStackingSVC = result_handler(stackingSVC, x_train, x_test, y_train, y_test, False)

            accuracyLogisticRegression = resultsLogisticRegression[0]
            recallLogisticRegression = resultsLogisticRegression[1]
            precisionMacroLogisticRegression = resultsLogisticRegression[2]
            f1LogisticRegression = resultsLogisticRegression[3]
            gMeanLogisticRegression = resultsLogisticRegression[4]
            accuracySVC = resultsSVC[0]
            recallSVC = resultsSVC[1]
            precisionMacroSVC = resultsSVC[2]
            f1SVC = resultsSVC[3]
            gMeanSVC = resultsSVC[4]
            accuracyStackingLogisticRegression = resultsStackingLogisticRegression[0]
            recallStackingLogisticRegression = resultsStackingLogisticRegression[1]
            precisionMacroStackingLogisticRegression = resultsStackingLogisticRegression[2]
            f1StackingLogisticRegression = resultsStackingLogisticRegression[3]
            gMeanStackingLogisticRegression = resultsStackingLogisticRegression[4]
            accuracyStackingSVC = resultsStackingSVC[0]
            recallStackingSVC = resultsStackingSVC[1]
            precisionMacroStackingSVC = resultsStackingSVC[2]
            f1StackingSVC = resultsStackingSVC[3]
            gMeanStackingSVC = resultsStackingSVC[4]

            result_printer(accuracyMultinomialNB,
                           recallMultinomialNB,
                           precisionMacroMultinomialNB,
                           f1MultinomialNB,
                           gMeanMultinomialNB,
                           "MultinomialNB ",
                           alpha,
                           None,
                           folds)
            result_printer(accuracyLogisticRegression,
                           recallLogisticRegression,
                           precisionMacroLogisticRegression,
                           f1LogisticRegression,
                           gMeanLogisticRegression,
                           "LogisticRegression ",
                           resultsLogisticRegression[5],
                           None,
                           1)
            result_printer(accuracySVC,
                           recallSVC,
                           precisionMacroSVC,
                           f1SVC,
                           gMeanSVC,
                           "SVC ",
                           resultsSVC[5],
                           None,
                           1)
            result_printer(accuracyStackingLogisticRegression,
                           recallStackingLogisticRegression,
                           precisionMacroStackingLogisticRegression,
                           f1StackingLogisticRegression,
                           gMeanStackingLogisticRegression,
                           "StackingLogisticRegression ",
                           resultsLogisticRegression[5],
                           resultsSVC[5],
                           1)
            result_printer(accuracyStackingSVC,
                           recallStackingSVC,
                           precisionMacroStackingSVC,
                           f1StackingSVC,
                           gMeanStackingSVC,
                           "StackingSVC ",
                           resultsSVC[5],
                           resultsLogisticRegression[5],
                           1)

            print("_______________________________")
            print(datetime.datetime.now().strftime("%D %H:%M:%S"))

    except BaseException as error:
        print("===================================================================================================")
        print("Error on_main: %s" % str(error))
        print("===================================================================================================")

