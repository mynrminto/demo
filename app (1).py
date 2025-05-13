
# ===============================================================
# Streamlit App  ── PECARN LSTM Predictor (Single‑Page Version)
# ===============================================================
# 変更点
#  • タブを廃止して 1 画面入力に統合
#  • 「月齢」は 0–23 を選択式 (selectbox) に変更
# ---------------------------------------------------------------

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path

# ───────────────────────────────────────────────────────────────
# 0) モデル読み込み
# ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str = "pecarn_lstm_model.keras"):
    if not Path(model_path).exists():
        st.error(f"モデルファイル {model_path} が見つかりません。")
        st.stop()
    return tf.keras.models.load_model(model_path)

model = load_model()

# ───────────────────────────────────────────────────────────────
# 1) 変数定義
# ───────────────────────────────────────────────────────────────
numeric_feature = "月齢"           # 数値扱いは月齢のみ
hema_loc_choices = ["なし", "前頭部", "後頭部", "頭頂部・側頭部"]

categorical_features = [
    "精神状態以外の神経学的異常",
    "Gender",
    "意識喪失",
    "傷害の重症度メカニズム",
    "痙攣発作",
    "親から見て様子がおかしい",
    "頭痛の重症度",
    "血腫の大きさ",
    "嘔吐",
    "鎖骨より上の外傷",
    "頭部以外の重大な外傷",
    "大泉門膨隆",
    "頭蓋骨骨折の兆候"
]

# ───────────────────────────────────────────────────────────────
# 2) カテゴリ→数値マッピング関数（変更なし）
# ───────────────────────────────────────────────────────────────
def map_feature_value(feature_name, selected_str):
    if feature_name == "精神状態以外の神経学的異常":
        return 0 if selected_str == "なし" else 1
    if feature_name == "Gender":
        return 1 if selected_str == "男児" else 2
    if feature_name == "意識喪失":
        return {"なし": 0, "あり": 1, "疑わしい": 2}[selected_str]
    if feature_name == "傷害の重症度メカニズム":
        return {"低": 0, "中": 1, "高": 2}[selected_str]
    if feature_name in {"痙攣発作", "親から見て様子がおかしい",
                        "鎖骨より上の外傷", "頭部以外の重大な外傷",
                        "大泉門膨隆", "頭蓋骨骨折の兆候"}:
        return 0 if selected_str == "なし" else 1
    if feature_name == "頭痛の重症度":
        return {"なし": 0, "軽度": 1, "中等度": 2, "高度": 3}[selected_str]
    if feature_name == "血腫の大きさ":
        return {"なし": 0, "<1cm": 1, "1-3cm": 2, ">3cm": 3}[selected_str]
    if feature_name == "嘔吐":
        return {"なし": 0, "1回": 1, "2回": 2, "3回以上": 3}[selected_str]
    return None  # 保険

# ───────────────────────────────────────────────────────────────
# 3) UI 生成（タブ無し、1 画面）
# ───────────────────────────────────────────────────────────────
st.title("中間リスク群　頭部 CT 撮像の要否予測 (PECARN‑LSTM)")
st.caption("月齢は 0–23 のプルダウン、他はカテゴリ選択で入力してください。")

user_inputs = {}

# ---------- 帳票レイアウト ----------
col1, col2, col3 = st.columns(3)

# 血腫の場所 (One‑Hot) は最上段にまとめる
with col1:
    hema_loc_selected = st.selectbox("血腫の場所 (One‑Hot)", hema_loc_choices)

hema_loc_dict = {   # one‑hot 初期化
    "HemaLoc_None": 0,
    "HemaLoc_Frontal": 0,
    "HemaLoc_Occipital": 0,
    "HemaLoc_ParietalTemporal": 0
}
hema_loc_dict.update({
    "HemaLoc_None": int(hema_loc_selected == "なし"),
    "HemaLoc_Frontal": int(hema_loc_selected == "前頭部"),
    "HemaLoc_Occipital": int(hema_loc_selected == "後頭部"),
    "HemaLoc_ParietalTemporal": int(hema_loc_selected == "頭頂部・側頭部")
})
user_inputs.update(hema_loc_dict)

# 月齢 (数値) は col2 内に配置
with col2:
    user_inputs[numeric_feature] = st.selectbox("月齢 (0–23)", list(range(24)))

# 残りカテゴリ変数を順に表示
feature_cols = st.columns(3)  # 3 列グリッド

for idx, feature in enumerate(categorical_features):
    with feature_cols[idx % 3]:
        # 選択肢を個別に定義
        choices = {
            "精神状態以外の神経学的異常": ["なし", "あり"],
            "Gender": ["男児", "女児"],
            "意識喪失": ["なし", "あり", "疑わしい"],
            "傷害の重症度メカニズム": ["低", "中", "高"],
            "痙攣発作": ["なし", "あり"],
            "親から見て様子がおかしい": ["なし", "あり"],
            "頭痛の重症度": ["なし", "軽度", "中等度", "高度"],
            "血腫の大きさ": ["なし", "<1cm", "1-3cm", ">3cm"],
            "嘔吐": ["なし", "1回", "2回", "3回以上"],
            "鎖骨より上の外傷": ["なし", "あり"],
            "頭部以外の重大な外傷": ["なし", "あり"],
            "大泉門膨隆": ["なし", "あり"],
            "頭蓋骨骨折の兆候": ["なし", "あり"]
        }[feature]

        choice = st.selectbox(feature, choices)
        user_inputs[feature] = map_feature_value(feature, choice)

# ------------------------------------------------------------
# 4) 予測処理
# ------------------------------------------------------------
if st.button("予測する"):
    # (a) DataFrame 化
    input_df = pd.DataFrame([user_inputs]).astype(float)

    # (b) LSTM 用に reshape
    x = input_df.to_numpy().reshape((1, 1, input_df.shape[1]))

    # (c) 推論
    y_prob = model.predict(x)[0][0]          # 1 = CT不要クラスの確率

    # (d) カットオフ (PECARN: 0.04)
    cutoff = 0.04
    need_ct = int(y_prob < cutoff)           # ★ 低確率ほど CT 推奨

    # (e) 結果メッセージ
    msg = "CT撮像をお勧めします" if need_ct else "CT撮像が必要な可能性は低いです"
    st.success(f"予測結果: {msg} ")

    # (f) 参考イメージ
    from PIL import Image
    img_name = "CT撮像.png" if need_ct else "経過観察.png"
    img_path = Path(__file__).parent / img_name
    if img_path.is_file():
        st.image(Image.open(img_path), use_column_width=True)
    else:
        st.warning(f"参考イメージ ({img_name}) が見つかりません。")
