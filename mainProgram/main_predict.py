import json
import os
import pandas as pd
from MLProcess.DrawPlot import DrawPlot
from MLProcess.Predict import Predict
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Scoring import Scoring
from userPackage.LoadDataset import LoadDataset
from userPackage.Package_CheckData import CheckData
from userPackage.Package_Encode import EncodeAllFeatures

MODEL_FOLDER_MAP = {
    "gbc": "gbc",
    "et": "et",
    "catboost": "catboost",
    "dt": "dt",
    "ada": "ada",
}

modelCutoffMap = {
    "ada": "bestCutoff_ada.json",
    "catboost": "bestCutoff_cb.json",
    "dt": "bestCutoff_dt.json",
    "et": "bestCutoff_et.json",
    "gbc": "bestCutoff_gbc.json",
}

modelNameList = list(MODEL_FOLDER_MAP.keys())

normalizeMethod = "Standard"
featureNum = 70
dataName = "AIP2025"

baseParamPath = f"../data/param/{normalizeMethod}/"
baseModelPath = "../data/finalModel/"
mlDataPath = "../data/mlData/"
mlScorePath = "../data/mlScore/"

os.makedirs(mlScorePath, exist_ok=True)

refParamPath = f"{baseParamPath}ada/"

ldObj = LoadDataset(minSeqLength=5)
indpNegPath = "../data/dataset/DS_Test2_neg.fasta"
indpPosPath = "../data/dataset/DS_Test2_pos.fasta"

indpNegSeqDict = ldObj.readFasta(indpNegPath)
indpPosSeqDict = ldObj.readFasta(indpPosPath)
testDict = {0: indpNegSeqDict, 1: indpPosSeqDict, -1: None}

encodeObj = EncodeAllFeatures()
encodeObj.dataEncodeSetup(
    saveFeatureDict=None,
    saveJsonPath=None,
    loadJsonPath=refParamPath + f"{dataName}_featureTypeDict.json",
    b_loadJson=True,
)
encodeIndpDf = encodeObj.dataEncodeOutPut(dataDict=testDict)

indpNmlzDf = encodeObj.dataNormalization(
    encodeIndpDf=encodeIndpDf,
    normalization=normalizeMethod.lower(),
    loadNmlzScalerPklPath=refParamPath + f"{dataName}_{normalizeMethod.lower()}Scaler.pkl",
    b_loadPkl=True,
)

skipFeatureList = [
    s for s in indpNmlzDf.columns if "MotifBitVec" in s
]

brtObj = encodeObj.dataBoruta(
    trainDf=None,
    runBoruta=False,
    featRankPath=mlDataPath + "Boruta-featureRank-XGB.csv",
    skipFeatureList=skipFeatureList,
)

encodeObj.dataDecidedFeatureNum(
    featureNum=featureNum,
    saveCsvPath=mlDataPath,
    indpDf=indpNmlzDf,
    brtObj=brtObj,
)

pycObj = PycaretWrapper()
setupDf = pycObj.doSetup(needTrain=False)

dataIndpDf = pd.read_csv(
    mlDataPath + f"/indp_F{featureNum}.csv", index_col=[0]
)
dataIndp_X = dataIndpDf.drop(["y"], axis=1)
dataIndp_y = dataIndpDf[["y"]]

finalModelList = pycObj.doLoadModel(
    path=baseModelPath,
    fileNameList=modelNameList,
    b_isFinalizedModel=True
)

predObjIndp = Predict(dataX=dataIndp_X, modelList=finalModelList)
predVectorListIndp, probVectorListIndp = predObjIndp.doPredict()

predVectorDf = pd.DataFrame(
    predVectorListIndp, index=modelNameList, columns=dataIndpDf.index
).T
probVectorDf = pd.DataFrame(
    probVectorListIndp, index=modelNameList, columns=dataIndpDf.index
).T

predVectorDf.to_csv(mlScorePath + "predVector.csv")
probVectorDf.to_csv(mlScorePath + "probVector.csv")

def doScoring(
        predVectorListIndp,
        probVectorListIndp,
        dataIndp_y,
        modelNameList,
        finalModelList,
        model_folder_map,
        baseParamPath,
        mlScorePath,
):
    scoreObjIndp = Scoring(
        predVectorList=predVectorListIndp,
        probVectorList=probVectorListIndp,
        answerDf=dataIndp_y,
        modelNameList=modelNameList,
    )

    for model_name in modelNameList:
        folder_name = model_folder_map[model_name]
        cutoff_file = modelCutoffMap[model_name]
        cutoff_path = f"{baseParamPath}{folder_name}/{cutoff_file}"

        if os.path.exists(cutoff_path):
            scoreObjIndp.loadCutoff(cutoff_path)
        else:
            print(f"[Warning] Cutoff file not found: {cutoff_path}")

    scoreDfIndp = scoreObjIndp.doScoring(
        b_optimizedMcc=True,
        sortColumn="mcc",
        path=mlScorePath + "singleModelScore.csv",
    )

    drawObj = DrawPlot(
        answerDf=dataIndp_y,
        modelList=finalModelList,
        modelNameList=modelNameList,
        predArrList=predVectorListIndp,
        probArrList=probVectorListIndp,
    )

    aucDf = drawObj.drawROC(
        colorList=None,
        title=False,
        save=True,
        saveLoc=mlScorePath + "singleModelIndeROC.png",
        show=True,
        dpi=300,
        figSize=(12, 9),
        topNum=len(modelNameList),
    )

    return scoreDfIndp


scoreDfIndp = doScoring(
    predVectorListIndp,
    probVectorListIndp,
    dataIndp_y,
    modelNameList,
    finalModelList,
    model_folder_map=MODEL_FOLDER_MAP,
    baseParamPath=baseParamPath,
    mlScorePath=mlScorePath,
)