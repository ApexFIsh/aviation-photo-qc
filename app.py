import os
import tempfile
import joblib
import pandas as pd
import streamlit as st
from PIL import Image
from features import extract_features


MODEL_PATH = "models/aviation_qc_model.pkl"


def make_comments(features):
    comments = []

    if features["highlight_clip_percent"] > 0.5:
        comments.append("高光溢出风险偏高：白色机身或天空可能有死白。")
    elif features["highlight_clip_percent"] > 0.1:
        comments.append("高光接近危险范围：不要再增加 Exposure / Whites / Highlights。")

    if features["shadow_clip_percent"] > 1.0:
        comments.append("暗部死黑偏多：机腹、起落架、发动机内部可能太黑。")

    if features["blue_red_ratio"] > 1.25:
        comments.append("蓝色通道偏强：可能天空过蓝，或者白色机身偏冷。建议检查 White Balance 白平衡。")

    if features["blue_green_ratio"] > 1.18:
        comments.append("蓝色相对绿色也偏强：天空或机身可能有偏蓝风险。")

    if features["saturation_mean"] > 95:
        comments.append("整体饱和度偏高：不建议继续增加 Saturation 饱和度。")

    if features["contrast"] < 38:
        comments.append("对比度可能偏低：画面可能 flat / 发灰。")

    if features["contrast"] > 75:
        comments.append("对比度可能偏强：高光和阴影反差可能太硬。")

    if features["sharpness_laplacian"] < 80:
        comments.append("锐度分数偏低：可能 soft / 发软。请放大检查注册号、窗线、机头边缘。")

    if not comments:
        comments.append("主要技术指标看起来正常，但仍需人工检查构图、热浪、尘点、压缩质量。")

    return comments


def camera_raw_recommendations(features):
    """
    Generate Adobe Camera Raw / Lightroom starting values.
    生成 Adobe Camera Raw / Lightroom 可直接参考的起始滑块参数。
    """

    settings = {
        "Exposure": 0.00,
        "Contrast": 0,
        "Highlights": 0,
        "Shadows": 0,
        "Whites": 0,
        "Blacks": 0,
        "Texture": 0,
        "Clarity": 0,
        "Dehaze": 0,
        "Vibrance": 0,
        "Saturation": 0,
        "Temperature": "0",
        "Tint": 0,
        "Color Mixer Blue Saturation": "0",
    }

    reasons = []

    # Highlight risk 高光风险
    if features["highlight_clip_percent"] > 0.5:
        settings["Exposure"] -= 0.10
        settings["Highlights"] -= 30
        settings["Whites"] -= 15
        reasons.append("高光溢出偏高：压低 Exposure / Highlights / Whites。")
    elif features["highlight_clip_percent"] > 0.1:
        settings["Exposure"] -= 0.05
        settings["Highlights"] -= 20
        settings["Whites"] -= 8
        reasons.append("高光接近危险范围：轻微压高光和白色。")

    # Shadow risk 暗部风险
    if features["shadow_clip_percent"] > 1.0:
        settings["Shadows"] += 20
        settings["Blacks"] += 5
        reasons.append("暗部死黑偏多：提高 Shadows，并稍微提高 Blacks。")
    elif features["shadow_clip_percent"] > 0.2:
        settings["Shadows"] += 10
        settings["Blacks"] += 3
        reasons.append("暗部略重：轻微提高 Shadows。")

    # Blue cast / cold white balance 蓝色偏强 / 白平衡偏冷
    if features["blue_red_ratio"] > 1.35 or features["blue_green_ratio"] > 1.18:
        settings["Temperature"] = "+2 to +5 warmer"
        settings["Saturation"] -= 3
        settings["Color Mixer Blue Saturation"] = "-5 to -10 if sky looks too blue"
        reasons.append("蓝色通道偏强：色温稍微加暖，必要时降低蓝色饱和度。")
    elif features["blue_red_ratio"] > 1.25:
        settings["Temperature"] = "+1 to +3 warmer"
        settings["Saturation"] -= 2
        reasons.append("蓝色略强：白平衡可以稍微加暖。")

    # Saturation 饱和度
    if features["saturation_mean"] > 105:
        settings["Vibrance"] -= 5
        settings["Saturation"] -= 7
        reasons.append("整体饱和度较高：降低 Saturation 和 Vibrance。")
    elif features["saturation_mean"] > 95:
        settings["Vibrance"] -= 2
        settings["Saturation"] -= 5
        reasons.append("整体饱和度偏高：降低 Saturation。")

    # Contrast 对比度
    if features["contrast"] < 38:
        settings["Contrast"] += 5
        settings["Clarity"] += 3
        settings["Dehaze"] += 2
        reasons.append("对比度偏低：增加 Contrast / Clarity / Dehaze。")
    elif features["contrast"] > 75:
        settings["Contrast"] -= 5
        settings["Highlights"] -= 10
        settings["Shadows"] += 5
        reasons.append("对比度偏强：降低 Contrast，并稍微压高光、提阴影。")

    # Sharpness / texture 锐度 / 纹理
    if features["sharpness_laplacian"] < 80:
        settings["Texture"] += 8
        settings["Clarity"] += 5
        reasons.append("锐度分数偏低：可以提高 Texture / Clarity，但要小心噪点和边缘光晕。")
    elif features["sharpness_laplacian"] < 150:
        settings["Texture"] += 5
        settings["Clarity"] += 3
        reasons.append("锐度一般：可以轻微提高 Texture / Clarity。")
    else:
        settings["Texture"] += 3
        settings["Clarity"] += 2
        reasons.append("锐度基础不错：只需要轻微 Texture / Clarity。")

    # Avoid over-editing 防止过度修图
    settings["Contrast"] = max(min(settings["Contrast"], 15), -15)
    settings["Highlights"] = max(min(settings["Highlights"], 0), -50)
    settings["Shadows"] = max(min(settings["Shadows"], 30), 0)
    settings["Whites"] = max(min(settings["Whites"], 5), -25)
    settings["Blacks"] = max(min(settings["Blacks"], 10), -5)
    settings["Texture"] = max(min(settings["Texture"], 12), 0)
    settings["Clarity"] = max(min(settings["Clarity"], 10), 0)
    settings["Dehaze"] = max(min(settings["Dehaze"], 5), 0)
    settings["Vibrance"] = max(min(settings["Vibrance"], 5), -10)
    settings["Saturation"] = max(min(settings["Saturation"], 3), -12)

    return settings, reasons


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

    features = extract_features(temp_path)

    with col2:
        st.subheader("Main Numbers 主要数据")

        st.metric("Brightness 平均亮度", f"{features['brightness_mean']:.2f}")
        st.metric("Contrast 对比度", f"{features['contrast']:.2f}")
        st.metric("Highlight clipping 高光溢出", f"{features['highlight_clip_percent']:.3f}%")
        st.metric("Shadow clipping 暗部死黑", f"{features['shadow_clip_percent']:.3f}%")
        st.metric("Saturation 饱和度", f"{features['saturation_mean']:.2f}")
        st.metric("Blue / Red 蓝红比例", f"{features['blue_red_ratio']:.3f}")
        st.metric("Blue / Green 蓝绿比例", f"{features['blue_green_ratio']:.3f}")
        st.metric("Sharpness 锐度分数", f"{features['sharpness_laplacian']:.2f}")

    st.subheader("AI Model Result AI 模型判断")

    if os.path.exists(MODEL_PATH):
        saved = joblib.load(MODEL_PATH)
        model = saved["model"]
        feature_columns = saved["feature_columns"]

        X_new = pd.DataFrame([features])[feature_columns]

        prediction = model.predict(X_new)[0]
        score = model.decision_function(X_new)[0]

        if prediction == 1:
            st.success(f"NORMAL / 接近好照片数据库。AI anomaly score: {score:.4f}")
        else:
            st.error(f"RISK / 和好照片数据库差异较大。AI anomaly score: {score:.4f}")
    else:
        st.warning("没有找到 AI 模型。请先运行 python train_model.py")

    st.subheader("Comments 建议")

    for comment in make_comments(features):
        st.write("- " + comment)

    st.subheader("Adobe Camera Raw Recommended Starting Values")
    st.write("下面这些是 Camera Raw / Lightroom 里的滑块起始建议，不是百分比。Exposure 是曝光档位，其他大多是 -100 到 +100 滑块值。")

    settings, reasons = camera_raw_recommendations(features)

    acr_rows = [
        {"Camera Raw Slider": "Exposure 曝光", "Recommended Value": format_exposure(settings["Exposure"])},
        {"Camera Raw Slider": "Contrast 对比度", "Recommended Value": format_signed_int(settings["Contrast"])},
        {"Camera Raw Slider": "Highlights 高光", "Recommended Value": format_signed_int(settings["Highlights"])},
        {"Camera Raw Slider": "Shadows 阴影", "Recommended Value": format_signed_int(settings["Shadows"])},
        {"Camera Raw Slider": "Whites 白色", "Recommended Value": format_signed_int(settings["Whites"])},
        {"Camera Raw Slider": "Blacks 黑色", "Recommended Value": format_signed_int(settings["Blacks"])},
        {"Camera Raw Slider": "Texture 纹理", "Recommended Value": format_signed_int(settings["Texture"])},
        {"Camera Raw Slider": "Clarity 清晰度", "Recommended Value": format_signed_int(settings["Clarity"])},
        {"Camera Raw Slider": "Dehaze 去朦胧", "Recommended Value": format_signed_int(settings["Dehaze"])},
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

    st.warning("这些数值是 Camera Raw 的起始点 starting point。最终还要看机身白色区域、注册号锐度、天空自然度和直方图。不要盲目全部照抄。")

    st.subheader("All Feature Data 全部数据")
    st.dataframe(pd.DataFrame([features]).T.rename(columns={0: "value"}))

    os.remove(temp_path)


if __name__ == "__main__":
    main()