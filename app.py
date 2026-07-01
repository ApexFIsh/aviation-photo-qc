import os
import tempfile

import joblib
import pandas as pd
import streamlit as st
from PIL import Image

from features import extract_features


MODEL_PATH = "models/aviation_qc_model.pkl"

# Stricter score threshold
# 分数低于这个值，即使模型没有判异常，也提示重点检查。
STRICT_SCORE_THRESHOLD = 0.03


def make_comments(features):
    comments = []

    if features["highlight_clip_percent"] > 0.5:
        comments.append("高光有明显风险：白色机身或天空可能接近死白。建议优先检查 Exposure 曝光 和 Whites 白色。")
    elif features["highlight_clip_percent"] > 0.15:
        comments.append("高光略亮：不要继续增加 Exposure 曝光 或 Whites 白色。")

    if features["shadow_clip_percent"] > 1.0:
        comments.append("暗部偏重：机腹、起落架、发动机内部可能偏黑。此版本不自动给 Shadows 阴影值，建议人工检查。")

    if features["blue_red_ratio"] > 1.55 and features["blue_green_ratio"] > 1.22:
        comments.append("蓝色通道明显偏强：可能天空过蓝，或白色机身偏冷。建议检查 White Balance 白平衡。")
    elif features["blue_red_ratio"] > 1.35 and features["blue_green_ratio"] > 1.18:
        comments.append("蓝色略强：这可能只是蓝天造成的，也可能是白色机身偏冷。请重点看机身白色区域。")

    if features["saturation_mean"] > 110:
        comments.append("整体饱和度偏高：建议降低 Saturation 饱和度 或 Blue Saturation 蓝色饱和度。")
    elif features["saturation_mean"] > 95:
        comments.append("饱和度略高：不建议继续增加 Saturation 饱和度。")

    if features["contrast"] < 35:
        comments.append("对比度偏低：画面可能 flat / 发灰。可以轻微增加 Contrast 对比度。")
    elif features["contrast"] > 70:
        comments.append("对比度偏强：反差可能太硬，白色和暗部容易不自然。")

    if features["sharpness_laplacian"] < 80:
        comments.append("锐度分数偏低：可能 soft / 发软。此版本不自动给 Texture / Clarity，建议放大检查注册号、窗线、机头边缘。")

    if not comments:
        comments.append("主要技术指标比较正常。Camera Raw 参数可以保持保守，不需要大幅调整。")

    return comments


def camera_raw_recommendations(features):
    """
    Conservative Adobe Camera Raw / Lightroom starting values.
    保守的 Camera Raw 起始参数。
    不输出 Highlights / Shadows / Texture / Clarity / Dehaze。
    """

    settings = {
        "Exposure": 0.00,
        "Contrast": 0,
        "Whites": 0,
        "Blacks": 0,
        "Vibrance": 0,
        "Saturation": 0,
        "Temperature": "0",
        "Tint": 0,
        "Color Mixer Blue Saturation": "0",
    }

    reasons = []

    brightness = features["brightness_mean"]
    contrast = features["contrast"]
    highlight_clip = features["highlight_clip_percent"]
    shadow_clip = features["shadow_clip_percent"]
    saturation = features["saturation_mean"]
    blue_red = features["blue_red_ratio"]
    blue_green = features["blue_green_ratio"]

    # Exposure / Whites 曝光和白色
    if highlight_clip > 0.75 or brightness > 175:
        settings["Exposure"] -= 0.10
        settings["Whites"] -= 8
        reasons.append("高光或整体亮度偏高：建议轻微降低 Exposure 曝光 和 Whites 白色。")
    elif highlight_clip > 0.20 or brightness > 162:
        settings["Exposure"] -= 0.05
        settings["Whites"] -= 5
        reasons.append("亮部略高：建议轻微压低 Exposure 曝光 或 Whites 白色。")
    elif brightness < 105:
        settings["Exposure"] += 0.05
        reasons.append("整体亮度偏低：可以小幅增加 Exposure 曝光。")

    # Contrast / Blacks 对比度和黑色
    if contrast < 35:
        settings["Contrast"] += 6
        settings["Blacks"] -= 2
        reasons.append("对比度偏低：轻微增加 Contrast 对比度，并略微压 Blacks 黑色。")
    elif contrast < 42:
        settings["Contrast"] += 3
        reasons.append("对比度略低：可以小幅增加 Contrast 对比度。")
    elif contrast > 72:
        settings["Contrast"] -= 5
        settings["Blacks"] += 2
        reasons.append("对比度偏强：轻微降低 Contrast 对比度，并让 Blacks 黑色不要太重。")

    # 暗部死黑只轻微影响 Blacks，不输出 Shadows。
    if shadow_clip > 1.0:
        settings["Blacks"] += 3
        reasons.append("暗部死黑偏多：不使用 Shadows 阴影滑块，只建议把 Blacks 黑色稍微抬一点。")

    # Saturation / Vibrance 饱和度
    if saturation > 115:
        settings["Saturation"] -= 6
        settings["Vibrance"] -= 3
        reasons.append("整体饱和度明显偏高：降低 Saturation 饱和度，并轻微降低 Vibrance 自然饱和度。")
    elif saturation > 100:
        settings["Saturation"] -= 3
        settings["Vibrance"] -= 1
        reasons.append("整体饱和度偏高：轻微降低 Saturation 饱和度。")
    elif saturation < 55:
        settings["Vibrance"] += 3
        reasons.append("整体饱和度偏低：可以小幅增加 Vibrance 自然饱和度。")

    # Blue cast / sky too blue 蓝色偏强
    if blue_red > 1.60 and blue_green > 1.25:
        settings["Temperature"] = "+2 to +4 warmer"
        settings["Color Mixer Blue Saturation"] = "-5 to -8 if sky looks too blue"
        settings["Saturation"] -= 2
        reasons.append("蓝色非常强：如果白色机身也偏蓝，色温向黄色方向加暖；如果只是天空太蓝，降低 Blue Saturation。")
    elif blue_red > 1.40 and blue_green > 1.18:
        settings["Temperature"] = "+1 to +3 warmer only if fuselage looks cold"
        settings["Color Mixer Blue Saturation"] = "-3 to -5 only if sky looks too blue"
        reasons.append("蓝色偏强但可能来自天空：不要盲目加暖，只在白色机身偏冷时调整 Temperature 色温。")

    # 防止过度修图
    settings["Exposure"] = max(min(settings["Exposure"], 0.10), -0.15)
    settings["Contrast"] = max(min(settings["Contrast"], 10), -10)
    settings["Whites"] = max(min(settings["Whites"], 3), -12)
    settings["Blacks"] = max(min(settings["Blacks"], 6), -5)
    settings["Vibrance"] = max(min(settings["Vibrance"], 5), -6)
    settings["Saturation"] = max(min(settings["Saturation"], 3), -8)

    if not reasons:
        reasons.append("这张图的数据比较正常：建议 Camera Raw 大部分滑块保持 0，只做人工微调。")

    return settings, reasons


def interpret_ai_score(score, final_prediction):
    """
    Make IsolationForest score easier to understand.
    更严格解释 AI 分数。
    分数越高，越像 good_photos 好照片数据库。
    """

    if final_prediction == -1:
        status = "RISK / 建议重点检查"
    else:
        status = "NORMAL / 可作为正常参考"

    if score >= 0.10:
        level = "非常接近好照片数据库"
        explanation = "这张图的数据非常像你的 good_photos 好照片数据库。"
    elif score >= 0.06:
        level = "比较接近好照片数据库"
        explanation = "这张图整体比较像你的好照片，通常问题不大。"
    elif score >= STRICT_SCORE_THRESHOLD:
        level = "基本接近好照片数据库"
        explanation = "这张图达到正常线，但仍建议检查颜色、曝光和锐度。"
    elif score >= 0.00:
        level = "边缘通过 / 需要人工复查"
        explanation = "这张图虽然没有明显偏离，但分数偏低，可能存在轻微问题。"
    elif score >= -0.03:
        level = "轻微偏离好照片数据库"
        explanation = "这张图和你的好照片标准有差异，建议认真检查曝光、颜色、对比度或锐度。"
    else:
        level = "明显偏离好照片数据库"
        explanation = "这张图和你的好照片标准差异较大，可能不适合直接提交。"

    return status, level, explanation


def format_signed_int(value):
    if isinstance(value, int):
        if value > 0:
            return f"+{value}"
        return str(value)
    return value


def format_exposure(value):
    if value > 0:
        return f"+{value:.2f}"
    return f"{value:.2f}"


def main():
    st.set_page_config(
        page_title="Aviation Photo QC Assistant",
        layout="wide"
    )

    st.title("Aviation Photo QC Assistant")
    st.write("航空照片质量检查助手：分析 exposure 曝光、contrast 对比度、color 颜色、saturation 饱和度、sharpness 锐度。")

    uploaded_file = st.file_uploader(
        "Upload one aviation photo 上传一张航空照片",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("请上传一张 JPG / PNG 航空照片。")
        return

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Photo 上传照片")
        st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name

    try:
        features = extract_features(temp_path)

        with col2:
            st.subheader("Main Numbers 主要数据")
            st.metric("Brightness 平均亮度", f"{features['brightness_mean']:.2f}")
            st.metric("Contrast 对比度", f"{features['contrast']:.2f}")
            st.metric("Highlight clipping 高光溢出检测", f"{features['highlight_clip_percent']:.3f}%")
            st.metric("Shadow clipping 暗部死黑检测", f"{features['shadow_clip_percent']:.3f}%")
            st.metric("Saturation 饱和度", f"{features['saturation_mean']:.2f}")
            st.metric("Blue / Red 蓝红比例", f"{features['blue_red_ratio']:.3f}")
            st.metric("Blue / Green 蓝绿比例", f"{features['blue_green_ratio']:.3f}")
            st.metric("Sharpness 锐度检测分数", f"{features['sharpness_laplacian']:.2f}")

        st.subheader("AI Database Similarity AI 数据库接近度")

        if os.path.exists(MODEL_PATH):
            saved = joblib.load(MODEL_PATH)
            model = saved["model"]
            feature_columns = saved["feature_columns"]

            X_new = pd.DataFrame([features])[feature_columns]

            prediction = model.predict(X_new)[0]
            score = model.decision_function(X_new)[0]

            # 更严格的最终判断
            if prediction == -1 or score < STRICT_SCORE_THRESHOLD:
                final_prediction = -1
            else:
                final_prediction = 1

            status, level, explanation = interpret_ai_score(score, final_prediction)

            if final_prediction == 1:
                st.success(f"AI 判断：{status}")
            else:
                st.error(f"AI 判断：{status}")

            st.write(f"**数据库接近度：{level}**")
            st.write(f"**技术分数：{score:.4f}**")
            st.caption("分数说明：越高越像你的 good_photos 好照片数据库；接近 0 表示边缘；负数表示偏离。")
            st.caption(explanation)

            training_photo_count = saved.get("training_photo_count", "Unknown")
            trained_at = saved.get("trained_at", "Unknown")
            model_note = saved.get("model_note", "Unknown")

            st.caption(f"Model training photos 训练照片数量: {training_photo_count}")
            st.caption(f"Model trained at 模型训练时间: {trained_at}")
            st.caption(f"Model note 模型说明: {model_note}")

        else:
            st.warning("没有找到 AI 模型。请先运行 python train_model.py")

        st.subheader("Comments 建议")
        for comment in make_comments(features):
            st.write("- " + comment)

        st.subheader("Adobe Camera Raw Recommended Starting Values")
        st.write("下面这些是 Camera Raw / Lightroom 里的保守起始建议。Exposure 是曝光档位，其他大多是 -100 到 +100 滑块值。")

        settings, reasons = camera_raw_recommendations(features)

        acr_rows = [
            {"Camera Raw Slider": "Exposure 曝光", "Recommended Value": format_exposure(settings["Exposure"])},
            {"Camera Raw Slider": "Contrast 对比度", "Recommended Value": format_signed_int(settings["Contrast"])},
            {"Camera Raw Slider": "Whites 白色", "Recommended Value": format_signed_int(settings["Whites"])},
            {"Camera Raw Slider": "Blacks 黑色", "Recommended Value": format_signed_int(settings["Blacks"])},
            {"Camera Raw Slider": "Vibrance 自然饱和度", "Recommended Value": format_signed_int(settings["Vibrance"])},
            {"Camera Raw Slider": "Saturation 饱和度", "Recommended Value": format_signed_int(settings["Saturation"])},
            {"Camera Raw Slider": "Temperature 色温", "Recommended Value": settings["Temperature"]},
            {"Camera Raw Slider": "Tint 色调", "Recommended Value": format_signed_int(settings["Tint"])},
            {"Camera Raw Slider": "Color Mixer / Blue Saturation 蓝色饱和度", "Recommended Value": settings["Color Mixer Blue Saturation"]},
        ]

        st.table(pd.DataFrame(acr_rows))

        st.subheader("Why these settings? 为什么这样调？")
        for reason in reasons:
            st.write("- " + reason)

        st.warning("这些数值只是 Camera Raw 的 starting point 起始点。最终还是要看机身白色、注册号锐度、天空自然度和直方图。不要盲目全部照抄。")

        st.subheader("All Feature Data 全部数据")
        st.dataframe(pd.DataFrame([features]).T.rename(columns={0: "value"}))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    main()