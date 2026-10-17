
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
    "gbc": ("GBC", "gbc"),
    "et": ("ET", "et"),
    "catboost": ("CB", "catboost"),
    "dt": ("DT", "dt"),
    "ada": ("AB", "ada"),
}

MODEL_NAMES = list(MODEL_CONFIG.keys())

DATA_NAME = "AIP2025"
DATASET_NAME = "Test1"
FEATURE_NUM = 70
NORMALIZE_METHOD = "Standard"

PARAM_PATH = "../data/param"
MODEL_PATH = "../data/finalModel"
ML_DATA_PATH = "../data/mlData"
ML_SCORE_PATH = "../data/mlScore"
DATASET_PATH = "../data/dataset"

os.makedirs(ML_SCORE_PATH, exist_ok=True)

REF_PARAM_PATH = os.path.join(PARAM_PATH, "ET")

FEATURE_DICT_PATH = os.path.join(
    REF_PARAM_PATH,
    f"{DATA_NAME}_featureTypeDict.json",
)

SCALER_PATH = os.path.join(
    REF_PARAM_PATH,
    f"{DATA_NAME}_{NORMALIZE_METHOD}Scaler.pkl",
)

FEATURE_RANK_PATH = os.path.join(
    ML_DATA_PATH,
    "Boruta-featureRank-XGB.csv",
)

NEG_DATA_PATH = os.path.join(
    DATASET_PATH, f"DS_{DATASET_NAME}_neg.fasta"
)
POS_DATA_PATH = os.path.join(
    DATASET_PATH, f"DS_{DATASET_NAME}_pos.fasta"
)

required_files = [
    FEATURE_DICT_PATH,
    SCALER_PATH,
    FEATURE_RANK_PATH,
    NEG_DATA_PATH,
    POS_DATA_PATH,
]

for model_name, (folder, _) in MODEL_CONFIG.items():
    required_files.extend([
        os.path.join(MODEL_PATH, folder),
        os.path.join(
            PARAM_PATH,
            folder,
            f"bestCutoff_{folder}.json",
        ),
    ])

for path in required_files:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到必要檔案：{path}")

loader = LoadDataset(minSeqLength=5)

test_data = {
    0: loader.readFasta(NEG_DATA_PATH),
    1: loader.readFasta(POS_DATA_PATH),
    -1: None,
}

encoder = EncodeAllFeatures()

encoder.dataEncodeSetup(
    saveFeatureDict=None,
    saveJsonPath=None,
    loadJsonPath=FEATURE_DICT_PATH,
    b_loadJson=True,
)

encoded_df = encoder.dataEncodeOutPut(
    dataDict=test_data
)

normalized_df = encoder.dataNormalization(
    encodeIndpDf=encoded_df,
    normalization=NORMALIZE_METHOD,
    loadNmlzScalerPklPath=SCALER_PATH,
    b_loadPkl=True,
)

skip_features = [
    col for col in normalized_df.columns
    if "MotifBitVec" in col
]

boruta = encoder.dataBoruta(
    trainDf=None,
    runBoruta=False,
    featRankPath=FEATURE_RANK_PATH,
    skipFeatureList=skip_features,
)

encoder.dataDecidedFeatureNum(
    featureNum=FEATURE_NUM,
    saveCsvPath=ML_DATA_PATH,
    indpDf=normalized_df,
    brtObj=boruta,
)

pycaret = PycaretWrapper()
pycaret.doSetup(needTrain=False)

indp_csv = os.path.join(
    ML_DATA_PATH,
    f"indp_F{FEATURE_NUM}.csv",
)

if not os.path.isfile(indp_csv):
    raise FileNotFoundError(
        f"找不到特徵 CSV：{indp_csv}"
    )

indp_df = pd.read_csv(
    indp_csv,
    index_col=0,
)

if "y" not in indp_df.columns:
    raise KeyError(
        f"{indp_csv} 缺少目標欄位 y"
    )

X = indp_df.drop(columns=["y"])
y = indp_df[["y"]]

models = []

for model_name, (folder, file_name) in MODEL_CONFIG.items():
    model_dir = os.path.join(MODEL_PATH, folder)

    loaded_models = pycaret.doLoadModel(
        path=model_dir,
        fileNameList=[file_name],
        b_isFinalizedModel=True,
    )

    if not loaded_models:
        raise RuntimeError(
            f"無法載入模型：{model_name}，路徑：{model_dir}"
        )

    models.append(loaded_models[0])

predictor = Predict(
    dataX=X,
    modelList=models,
)

predictions, probabilities = predictor.doPredict()

if (
    len(predictions) != len(MODEL_NAMES)
    or len(probabilities) != len(MODEL_NAMES)
):
    raise ValueError("預測結果數量與模型數量不一致")

pred_df = pd.DataFrame(
    predictions,
    index=MODEL_NAMES,
    columns=indp_df.index,
).T

prob_df = pd.DataFrame(
    probabilities,
    index=MODEL_NAMES,
    columns=indp_df.index,
).T

pred_df.to_csv(
    os.path.join(ML_SCORE_PATH, "predVector.csv")
)

prob_df.to_csv(
    os.path.join(ML_SCORE_PATH, "probVector.csv")
)

cutoffs = {}

for model_name, (folder, file_name) in MODEL_CONFIG.items():
    cutoff_path = os.path.join(
        PARAM_PATH,
        folder,
        f"bestCutoff_{folder}.json",
    )

    with open(cutoff_path, "r", encoding="utf-8") as f:
        cutoff_data = json.load(f)

    if not isinstance(cutoff_data, dict):
        raise TypeError(
            f"Cutoff JSON 格式不是字典：{cutoff_path}"
        )

    cutoff_key = next(
        (
            key for key in (model_name, folder, file_name)
            if key in cutoff_data
        ),
        None,
    )

    if cutoff_key is None:
        raise KeyError(
            f"{cutoff_path} 找不到 {model_name} 的 cutoff；"
            f"JSON keys：{list(cutoff_data.keys())}"
        )

    cutoff = cutoff_data[cutoff_key]

    if isinstance(cutoff, bool) or not isinstance(
        cutoff, (int, float)
    ):
        raise TypeError(
            f"{model_name} 的 cutoff 不是數值：{cutoff}"
        )

    if not 0 <= cutoff <= 1:
        raise ValueError(
            f"{model_name} 的 cutoff 超出 [0, 1]：{cutoff}"
        )

    cutoffs[model_name] = cutoff
    print(f"載入 cutoff：{model_name} = {cutoff}")

cutoff_output_path = os.path.join(
    ML_SCORE_PATH,
    "merged_bestCutoff.json",
)

with open(
    cutoff_output_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        cutoffs,
        f,
        ensure_ascii=False,
        indent=4,
    )

scorer = Scoring(
    predVectorList=predictions,
    probVectorList=probabilities,
    answerDf=y,
    modelNameList=MODEL_NAMES,
)

scorer.loadCutoff(cutoff_output_path)

score_df = scorer.doScoring(
    b_optimizedMcc=True,
    sortColumn="mcc",
    path=os.path.join(
        ML_SCORE_PATH,
        "singleModelScore.csv",
    ),
)

plotter = DrawPlot(
    answerDf=y,
    modelList=models,
    modelNameList=MODEL_NAMES,
    predArrList=predictions,
    probArrList=probabilities,
)

plotter.drawROC(
    colorList=None,
    title=False,
    save=True,
    saveLoc=os.path.join(
        ML_SCORE_PATH,
        "singleModelIndeROC.png",
    ),
    show=True,
    dpi=300,
    figSize=(12, 9),
    topNum=len(MODEL_NAMES),
)

print(f"\n資料集：{DATASET_NAME}")
print(f"特徵數量：{FEATURE_NUM}")
print("\n評分完成！")
print(score_df)