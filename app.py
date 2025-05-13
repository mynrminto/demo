
import streamlit as st
import numpy as np
import pickle
from sklearn.linear_model import Ridge

def main():
    st.title("ΔCDAI 3M Prediction App")
    st.write("このアプリは複数の臨床指標を入力し、リッジ回帰モデルで3ヶ月後のΔCDAIの予測を行います。")

    # ----------------------------------------
    # 1. 基本入力（連続値・カテゴリ変数）
    # ----------------------------------------
    Sex = st.selectbox("Sex (女性=1, 男性=0)", [0, 1])
    Age = st.number_input("Age (整数)", min_value=0, max_value=120, value=50, step=1)
    drug_line = st.number_input("何剤目 (整数)", min_value=1, value=1, step=1)
    Bio_naive = st.selectbox("Bio/JAK naïve (+ = 0, - = 1)", [0, 1])
    CDAI = st.number_input("CDAI (整数)", min_value=0, value=10, step=1)
    CRP = st.number_input("CRP (小数点以下2桁まで)", min_value=0.00, value=0.00, step=0.01, format="%.2f")
    ESR = st.number_input("ESR (整数)", min_value=0, value=10, step=1)
    RF = st.number_input("RF (整数)", min_value=0, value=30, step=1)
    MMP_3 = st.number_input("MMP-3 (整数)", min_value=0, value=100, step=1)
    ACPA = st.number_input("ACPA (整数)", min_value=0, value=1, step=1)
    TJC = st.number_input("TJC (整数)", min_value=0, value=5, step=1)
    SJC = st.number_input("SJC (整数)", min_value=0, value=5, step=1)
    Pt_VAS = st.number_input("Pt VAS (整数)", min_value=0, value=50, step=1)
    Dr_VAS = st.number_input("Dr VAS (整数)", min_value=0, value=50, step=1)
    PSL = st.selectbox("PSL (使用有無)", [0, 1])
    HAQ = st.number_input("HAQ (整数)", min_value=0, value=0, step=1)
    Stage = st.selectbox("Stage", [1, 2, 3, 4])
    Class = st.selectbox("Class", [1, 2, 3, 4])
    csDMARDs = st.selectbox("csDMARDs (併用数)", [0, 1, 2, 3])

    # ----------------------------------------
    # 2. MTX, SASP, IGU, BUC, TAC, MZ の使用有無（チェックボックス）
    # ----------------------------------------
    MTX_checkbox = st.checkbox("MTX（使用ならチェック）", value=False)
    MTX = 1 if MTX_checkbox else 0

    SASP_checkbox = st.checkbox("SASP（使用ならチェック）", value=False)
    SASP = 1 if SASP_checkbox else 0

    IGU_checkbox = st.checkbox("IGU（使用ならチェック）", value=False)
    IGU = 1 if IGU_checkbox else 0

    BUC_checkbox = st.checkbox("BUC（使用ならチェック）", value=False)
    BUC = 1 if BUC_checkbox else 0

    TAC_checkbox = st.checkbox("TAC（使用ならチェック）", value=False)
    TAC = 1 if TAC_checkbox else 0

    MZ_checkbox = st.checkbox("MZ（使用ならチェック）", value=False)
    MZ = 1 if MZ_checkbox else 0

    # ----------------------------------------
    # 3. バイオ薬（1種類のみ選択 → 選ばれたもの=1、それ以外=0）
    # ----------------------------------------
    bio_options = [
        "None",  # 未使用を入れたい場合
        "ABT", "ADA", "BAR", "CZP", "ETN",
        "FIL", "GLM", "IFX", "Pef", "Sar",
        "TCZ", "TOF", "UPA"
    ]
    selected_bio = st.selectbox("使用Bio薬を一つ選択", bio_options)

    使用Bio_ABT = 1 if selected_bio == "ABT" else 0
    使用Bio_ADA = 1 if selected_bio == "ADA" else 0
    使用Bio_BAR = 1 if selected_bio == "BAR" else 0
    使用Bio_CZP = 1 if selected_bio == "CZP" else 0
    使用Bio_ETN = 1 if selected_bio == "ETN" else 0
    使用Bio_FIL = 1 if selected_bio == "FIL" else 0
    使用Bio_GLM = 1 if selected_bio == "GLM" else 0
    使用Bio_IFX = 1 if selected_bio == "IFX" else 0
    使用Bio_Pef = 1 if selected_bio == "Pef" else 0
    使用Bio_Sar = 1 if selected_bio == "Sar" else 0
    使用Bio_TCZ = 1 if selected_bio == "TCZ" else 0
    使用Bio_TOF = 1 if selected_bio == "TOF" else 0
    使用Bio_UPA = 1 if selected_bio == "UPA" else 0

    # ----------------------------------------
    # 4. 入力ベクトルの作成
    #    （学習時の順序に合わせること）
    # ----------------------------------------
    input_data = np.array([
        Sex, Age, drug_line, Bio_naive, CDAI, CRP, ESR, RF, MMP_3, ACPA,
        TJC, SJC, Pt_VAS, Dr_VAS, PSL, HAQ, Stage, Class, csDMARDs,
        MTX, SASP, IGU, BUC, TAC, MZ,
        使用Bio_ABT, 使用Bio_ADA, 使用Bio_BAR, 使用Bio_CZP, 使用Bio_ETN,
        使用Bio_FIL, 使用Bio_GLM, 使用Bio_IFX, 使用Bio_Pef, 使用Bio_Sar,
        使用Bio_TCZ, 使用Bio_TOF, 使用Bio_UPA
    ]).reshape(1, -1)

    # ----------------------------------------
    # 5. 学習済みモデルの読み込み
    # ----------------------------------------
    try:
        with open("ridge_model.pkl", "rb") as f:
            loaded_model = pickle.load(f)
        st.success("学習済みリッジ回帰モデルを読み込みました。")
    except FileNotFoundError:
        st.warning("モデルファイル (ridge_model.pkl) が見つかりません。ダミーモデルを使用します。")
        loaded_model = Ridge(alpha=1.0)
        X_dummy = np.random.rand(10, input_data.shape[1])
        y_dummy = np.random.rand(10)
        loaded_model.fit(X_dummy, y_dummy)

    # ----------------------------------------
    # 6. 予測と表示
    # ----------------------------------------
    if st.button("Predict"):
        prediction = loaded_model.predict(input_data)
        st.write(f"予測されたΔCDAI: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()
