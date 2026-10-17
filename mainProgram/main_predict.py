
import json
import os
import pandas as pd

from MLProcess.DrawPlot import DrawPlot
from MLProcess.Predict import Predict
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Scoring import Scoring
from userPackage.LoadDataset import LoadDataset
from userPackage.Package_Encode import EncodeAllFeatures

MODEL_CONFIG = {
    "gbc": {
        "folder": "GBC",
        "file": "gbc",
    },
    "et": {
        "folder": "ET",
        "file": "et",
    },
    "catboost": {
        "folder": "CB",
        "file": "catboost",
    },
    "dt": {
        "folder": "DT",
        "file": "dt",
    },
    "ada": {
        "folder": "AB",
        "file": "ada",
    },
}

modelNameList = list(MODEL_CONFIG.keys())

normalizeMethod = "Standard"
featureNum = 70
dataName = "AIP2025"

baseParamPath = "../data/param/"
baseModelPath = "../data/finalModel/"
mlDataPath = "../data/mlData/"
featureStatPath = "../data/featureStat/"
mlScorePath = "../data/mlScore/"

os.makedirs(mlScorePath, exist_ok=True)

# 使用 ET 資料夾的前處理參數。
# 只有在五個模型原本共用相同前處理設定時，這樣才適用。
refParamPath = os.path.join(baseParamPath, "ET")

featureTypeDictPath = os.path.join(
    refParamPath,
    f"{dataName}_featureTypeDict.json",
)

scalerPath = os.path.join(
    refParamPath,
    f"{dataName}_{normalizeMethod}Scaler.pkl",
)

requiredFiles = [
    featureTypeDictPath,
    scalerPath,
    os.path.join(mlDataPath, "Boruta-featureRank-XGB.csv"),
    "../data/dataset/DS_Test1_neg.fasta",
    "../data/dataset/DS_Test1_pos.fasta",
]

for filePath in requiredFiles:
    if not os.path.isfile(filePath):
        raise FileNotFoundError(
            f"找不到必要檔案：{filePath}"
        )

for modelName, config in MODEL_CONFIG.items():
    folderName = config["folder"]

    modelDir = os.path.join(baseModelPath, folderName)
    cutoffPath = os.path.join(
        baseParamPath,
        folderName,
        f"bestCutoff_{folderName}.json",
    )

    if not os.path.isdir(modelDir):
        raise FileNotFoundError(
            f"找不到模型資料夾：{modelDir}"
        )

    if not os.path.isfile(cutoffPath):
        raise FileNotFoundError(
            f"找不到模型 cutoff 檔案：{cutoffPath}"
        )
ldObj = LoadDataset(minSeqLength=5)

indpNegPath = "../data/dataset/DS_Test2_neg.fasta"
indpPosPath = "../data/dataset/DS_Test2_pos.fasta"

indpNegSeqDict = ldObj.readFasta(indpNegPath)
indpPosSeqDict = ldObj.readFasta(indpPosPath)

testDict = {
    0: indpNegSeqDict,
    1: indpPosSeqDict,
    -1: None,
}

encodeObj = EncodeAllFeatures()

encodeObj.dataEncodeSetup(
    saveFeatureDict=None,
    saveJsonPath=None,
    loadJsonPath=featureTypeDictPath,
    b_loadJson=True,
)

encodeIndpDf = encodeObj.dataEncodeOutPut(
    dataDict=testDict
)

indpNmlzDf = encodeObj.dataNormalization(
    encodeIndpDf=encodeIndpDf,
    normalization=normalizeMethod,
    loadNmlzScalerPklPath=scalerPath,
    b_loadPkl=True,
)

skipFeatureList = [
    column
    for column in indpNmlzDf.columns
    if "MotifBitVec" in column
]

brtObj = encodeObj.dataBoruta(
    trainDf=None,
    runBoruta=False,
    featRankPath=os.path.join(
        mlDataPath,
        "Boruta-featureRank-XGB.csv",
    ),
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

indpCsvPath = os.path.join(
    mlDataPath,
    f"indp_F{featureNum}.csv",
)

if not os.path.isfile(indpCsvPath):
    raise FileNotFoundError(
        f"找不到特徵 CSV：{indpCsvPath}"
    )

dataIndpDf = pd.read_csv(
    indpCsvPath,
    index_col=0,
)

if "y" not in dataIndpDf.columns:
    raise KeyError(
        f"{indpCsvPath} 缺少目標欄位 y"
    )

dataIndp_X = dataIndpDf.drop(columns=["y"])
dataIndp_y = dataIndpDf[["y"]]

finalModelList = []

for modelName in modelNameList:
    config = MODEL_CONFIG[modelName]

    modelDir = os.path.join(
        baseModelPath,
        config["folder"],
    )

    loadedModels = pycObj.doLoadModel(
        path=modelDir,
        fileNameList=[config["file"]],
        b_isFinalizedModel=True,
    )

    if not loadedModels:
        raise RuntimeError(
            f"無法載入模型：{modelName}，路徑：{modelDir}"
        )

    finalModelList.append(loadedModels[0])

predObjIndp = Predict(
    dataX=dataIndp_X,
    modelList=finalModelList,
)

predVectorListIndp, probVectorListIndp = (
    predObjIndp.doPredict()
)

if len(predVectorListIndp) != len(modelNameList):
    raise ValueError(
        "預測模型數量與 modelNameList 不一致。"
    )

if len(probVectorListIndp) != len(modelNameList):
    raise ValueError(
        "機率預測數量與 modelNameList 不一致。"
    )

predVectorDf = pd.DataFrame(
    predVectorListIndp,
    index=modelNameList,
    columns=dataIndpDf.index,
).T

probVectorDf = pd.DataFrame(
    probVectorListIndp,
    index=modelNameList,
    columns=dataIndpDf.index,
).T

predVectorDf.to_csv(
    os.path.join(mlScorePath, "predVector.csv")
)

probVectorDf.to_csv(
    os.path.join(mlScorePath, "probVector.csv")
)
def loadModelCutoffs(modelNameList, modelConfig, baseParamPath):
    combinedCutoffDict = {}

    for modelName in modelNameList:
        config = modelConfig[modelName]
        folderName = config["folder"]

        cutoffPath = os.path.join(
            baseParamPath,
            folderName,
            f"bestCutoff_{folderName}.json",
        )

        with open(
            cutoffPath,
            "r",
            encoding="utf-8",
        ) as file:
            cutoffData = json.load(file)

        if not isinstance(cutoffData, dict):
            raise TypeError(
                f"Cutoff JSON 格式不是字典：{cutoffPath}"
            )

        possibleKeys = [
            modelName,
            folderName,
            config["file"],
        ]

        matchedKey = next(
            (
                key
                for key in possibleKeys
                if key in cutoffData
            ),
            None,
        )

        if matchedKey is None:
            raise KeyError(
                f"{cutoffPath} 找不到模型 {modelName} 的 cutoff。"
                f"JSON keys：{list(cutoffData.keys())}"
            )

        cutoffValue = cutoffData[matchedKey]

        if not isinstance(cutoffValue, (int, float)):
            raise TypeError(
                f"{cutoffPath} 的 cutoff 不是數值：{cutoffValue}"
            )

        if not 0 <= cutoffValue <= 1:
            raise ValueError(
                f"{cutoffPath} 的 cutoff 超出 [0, 1]：{cutoffValue}"
            )

        combinedCutoffDict[modelName] = cutoffValue

        print(
            f"載入 cutoff：{modelName} = {cutoffValue}"
        )

    return combinedCutoffDict
def doScoringSelectedModels(
    predVectorListIndp,
    probVectorListIndp,
    dataIndp_y,
    modelNameList,
    finalModelList,
    modelConfig,
    baseParamPath,
    mlScorePath,
):
    scoreObjIndp = Scoring(
        predVectorList=predVectorListIndp,
        probVectorList=probVectorListIndp,
        answerDf=dataIndp_y,
        modelNameList=modelNameList,
    )
    combinedCutoffDict = loadModelCutoffs(
        modelNameList=modelNameList,
        modelConfig=modelConfig,
        baseParamPath=baseParamPath,
    )
    mergedCutoffPath = os.path.join(
        mlScorePath,
        "merged_bestCutoff.json",
    )
    with open(
        mergedCutoffPath,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            combinedCutoffDict,
            file,
            ensure_ascii=False,
            indent=4,
        )
    scoreObjIndp.loadCutoff(mergedCutoffPath)
    scoreDfIndp = scoreObjIndp.doScoring(
        b_optimizedMcc=True,
        sortColumn="mcc",
        path=os.path.join(
            mlScorePath,
            "singleModelScore.csv",
        ),
    )
    drawObj = DrawPlot(
        answerDf=dataIndp_y,
        modelList=finalModelList,
        modelNameList=modelNameList,
        predArrList=predVectorListIndp,
        probArrList=probVectorListIndp,
    )
    drawObj.drawROC(
        colorList=None,
        title=False,
        save=True,
        saveLoc=os.path.join(
            mlScorePath,
            "singleModelIndeROC.png",
        ),
        show=True,
        dpi=300,
        figSize=(12, 9),
        topNum=len(modelNameList),
    )
    return scoreDfIndp
scoreDfIndp = doScoringSelectedModels(
    predVectorListIndp=predVectorListIndp,
    probVectorListIndp=probVectorListIndp,
    dataIndp_y=dataIndp_y,
    modelNameList=modelNameList,
    finalModelList=finalModelList,
    modelConfig=MODEL_CONFIG,
    baseParamPath=baseParamPath,
    mlScorePath=mlScorePath,
)