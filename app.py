import os  # 導入操作系統相關功能的模組
import numpy as np  # 導入 NumPy，用於數值計算
import joblib  # 導入 joblib，用於模型和數據的序列化
from flask import Flask, render_template, request  # 導入 Flask 相關模組
import tensorflow as tf  # 導入 TensorFlow 用於深度學習模型
from tensorflow import keras  # 導入 Keras，用於加載深度學習模型
import logging  # 導入 logging，用於記錄日誌

# 設置日誌級別為 DEBUG
logging.basicConfig(level=logging.DEBUG)

# 初始化 Flask 應用
app = Flask(__name__, template_folder='templates')

# 獲取當前文件夾的絕對路徑
basedir = os.path.abspath(os.path.dirname(__file__))

# 定義模型和標準化器的檔案路徑
model_path = os.path.join(basedir, 'heart_disease_model.keras')
scaler_path = os.path.join(basedir, 'scaler.pkl')

# 定義 focal loss 函數
def focal_loss_fixed(y_true, y_pred, gamma=2.0, alpha=0.25):
    """自定義 focal loss 函數，用於處理不平衡資料集"""
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    fl = -alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(tf.clip_by_value(p_t, 1e-8, 1.0))
    return tf.reduce_mean(fl)

# 註冊 focal loss 函數，讓 Keras 可以識別
keras.utils.get_custom_objects()['focal_loss_fixed'] = focal_loss_fixed

# 初始化模型和標準化器
model = None
scaler = None

# 嘗試加載模型和標準化器
try:
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    logging.info("模型和標準化器成功加載")
except Exception as e:
    logging.error(f"加載模型或標準化器時發生錯誤: {str(e)}")

# 定義模型需要的特徵欄位
features = [
    'age', 'male', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
    'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
    'diaBP', 'BMI', 'heartRate', 'glucose'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    """處理首頁請求"""
    # 初始化預測結果和風險等級
    prediction_result = None
    risk_level = None

    if request.method == 'POST':
        logging.info("接收到 POST 請求")
        try:
            # 收集從表單獲取的輸入數據，若無輸入則預設為 0
            input_data = [float(request.form.get(feature, 0)) for feature in features]
            logging.info(f"輸入數據：{input_data}")

            # 檢查模型和標準化器是否已成功加載
            if model is not None and scaler is not None:
                # 檢查輸入數據的長度是否符合模型的要求
                if len(input_data) == len(features):
                    # 將輸入數據轉為 NumPy 陣列並重塑為二維陣列
                    input_array = np.array(input_data).reshape(1, -1)
                    
                    # 使用標準化器對輸入數據進行標準化
                    input_scaled = scaler.transform(input_array)
                    
                    # 使用模型進行預測
                    prediction = model.predict(input_scaled)
                    
                    # 解析預測結果
                    if len(prediction.shape) == 2 and prediction.shape[1] > 1:
                        prediction_probability = prediction[0][1]  # 取得第二類的機率
                    else:
                        prediction_probability = prediction[0]  # 單輸出時直接取得預測值
                    
                    # 將預測結果轉換為百分比並四捨五入
                    prediction_result = round(prediction_probability * 100, 2)

                    # 根據預測結果設置風險等級
                    if prediction_result < 10:
                        risk_level = "低風險"
                    elif prediction_result < 20:
                        risk_level = "中等風險"
                    else:
                        risk_level = "高風險"
                    
                    logging.info(f"預測結果：{prediction_result}%, 風險等級：{risk_level}")
                else:
                    logging.error("輸入資料長度不符")
                    raise ValueError("輸入資料長度與預期不符")
            else:
                logging.error("模型或標準化器未加載")
        except Exception as e:
            logging.error(f"預測過程中發生錯誤：{str(e)}")
            prediction_result = None
            risk_level = None

    # 渲染 index.html 模板並傳遞預測結果和風險等級
    return render_template(
        'index.html',
        features=features,
        prediction=prediction_result,
        risk_level=risk_level
    )

if __name__ == '__main__':
    logging.info("啟動 Flask 應用程序")
    # 啟動 Flask 伺服器
    app.run(debug=True, host='0.0.0.0', port=5000)

