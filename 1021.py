# 導入所需的庫
from sklearn.preprocessing import StandardScaler  # 用於標準化數據
from sklearn.model_selection import KFold  # 用於K折交叉驗證
from imblearn.over_sampling import SMOTE  # 用於過取樣的SMOTE方法
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # 用於模型訓練的回調函數
from sklearn.metrics import classification_report, confusion_matrix  # 用於生成分類報告和混淆矩陣
import matplotlib.pyplot as plt  # 用於繪圖
import seaborn as sns  # 用於繪製混淆矩陣
import tensorflow as tf  # 用於深度學習
import numpy as np  # 用於數據處理
import pandas as pd  # 用於數據處理
from matplotlib import font_manager  # 用於設定中文字型
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, auc

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# 設定中文字型，這裡以SimHei為例
plt.rcParams['font.family'] = 'SimHei'  # 使用SimHei字型

# 確保結果可重現
np.random.seed(42)
tf.random.set_seed(42)
def balance_data(data, target_column):
    """
    平衡資料，透過減少多數類別樣本進行欠抽樣。
    
    Args:
        data (pd.DataFrame): 數據集
        target_column (str): 目標欄位名稱
        
    Returns:
        pd.DataFrame: 平衡後的數據集
    """
    # 分離多數類別與少數類別
    positive = data[data[target_column] == 1]
    negative = data[data[target_column] == 0]
    
    # 取得較少的類別的樣本數
    min_samples = min(len(positive), len(negative))
    
    # 隨機抽樣多數類別，使其樣本數等於少數類別
    if len(positive) > len(negative):
        positive = positive.sample(n=min_samples, random_state=42)
    else:
        negative = negative.sample(n=min_samples, random_state=42)
    
    # 合併平衡後的數據
    balanced_data = pd.concat([positive, negative])
    
    # 打亂數據順序
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_data

def 載入與預處理數據(文件路徑):
    """
    載入並預處理數據，包括處理缺失值和分離特徵與標籤。
    :param 文件路徑: 數據文件的路徑 (Excel格式)
    :return: 特徵數據X、標籤y以及特徵名稱列表
    """
    # 從Excel文件中載入數據
    數據框 = pd.read_excel(文件路徑)
    
    # 使用balance_data函數平衡數據
    數據框 = balance_data(數據框, "TenYearCHD")
    
    # 定義需要的特徵列
    特徵列 = ['age', 'male', 'education', 'currentSmoker', 'cigsPerDay', 
            'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 
            'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    
    # 處理缺失值：使用中位數加上隨機噪聲填補
    for 特徵 in 特徵列:
        if 數據框[特徵].isnull().sum() > 0:
            中位數 = 數據框[特徵].median()
            標準差 = 數據框[特徵].std()
            數據框[特徵].fillna(中位數 + np.random.normal(0, 0.1 * 標準差), inplace=True)
    
    # 獲取特徵和標籤數據
    X = 數據框[特徵列].values.astype(np.float64)
    y = 數據框["TenYearCHD"].values.astype(np.int8)
    
    return X, y, 特徵列
    # 查看數據的前幾行和基本資訊
    data_info = data.info()
    data_head = data.head()
    data_target_distribution = data['TenYearCHD'].value_counts()  # 查看目標列分佈

    data_info, data_head, data_target_distribution

def focal_loss(γ=5., α=0.25):  # 調整α值
    def loss(y_true, y_pred):
        # 更平衡的損失函數計算
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_mean(
            α * tf.pow(1. - pt_1, γ) * tf.math.log(pt_1 + 1e-7)
        ) - tf.reduce_mean(
            (1 - α) * tf.pow(pt_0, γ) * tf.math.log(1. - pt_0 + 1e-7)
        )
    return loss

def 建立模型(輸入維度):
    """
    建立神經網絡模型
    :param 輸入維度: 輸入層的特徵數量
    :return: 建立好的模型
    """
    模型 = tf.keras.models.Sequential([  # 使用Sequential模型
        tf.keras.layers.Dense(1024, activation='relu', input_shape=(輸入維度,)),  # 第一層，1024個神經元
        tf.keras.layers.BatchNormalization(),  # 批標準化層
        tf.keras.layers.Dropout(0.6),  # Dropout層，防止過擬合
        tf.keras.layers.Dense(512, activation='relu'),  # 第二層，512個神經元
        tf.keras.layers.BatchNormalization(),  # 批標準化層
        tf.keras.layers.Dropout(0.5),  # Dropout層
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # 第三層，256個神經元，L2正則化
        tf.keras.layers.BatchNormalization(),  # 批標準化層
        tf.keras.layers.Dropout(0.4),  # Dropout層
        tf.keras.layers.Dense(2, activation='softmax')  # 輸出層，2個神經元 (二分類問題)
    ])
    
    # 編譯模型
    模型.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # 使用Adam優化器
        loss=focal_loss(),  # 使用焦點損失函數
        metrics=['accuracy', 'precision', 'recall', tf.keras.metrics.AUC()]  # 設置評估指標
    )
    
    return 模型  # 返回模型


def 繪製混淆矩陣(真實標籤, 預測標籤, 折數=None):
    """
    繪製混淆矩陣圖。
    :param 真實標籤: 驗證集的真實標籤
    :param 預測標籤: 驗證集的預測標籤
    :param 折數: 當前是第幾折
    """
    # 定義類別名稱
    類別名稱 = ['無心臟病', '有心臟病']
    
    # 計算混淆矩陣
    混淆矩陣 = confusion_matrix(真實標籤, 預測標籤)
    
    # 設置圖表大小
    plt.figure(figsize=(6, 5))
    
    # 繪製熱圖
    sns.heatmap(混淆矩陣, annot=True, fmt='d', cmap='Blues', xticklabels=類別名稱, yticklabels=類別名稱)
    
    # 設置標題
    標題 = '混淆矩陣'
    if 折數 is not None:
        標題 += f' (第{折數}折)'  # 顯示第幾折
    plt.title(標題, fontsize=14)
    
    # 添加X和Y軸標籤
    plt.xlabel('預測標籤', fontsize=12)
    plt.ylabel('真實標籤', fontsize=12)
    
    # 顯示圖表
    plt.show()
    

def 詳細評估(真實標籤, 預測標籤):

    
    # 分類報告
    print(classification_report(真實標籤, 預測標籤))
    
    # ROC曲線
    fpr, tpr, _ = roc_curve(真實標籤, 預測標籤)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
def plot_training_history(history, fold_num=None):
    """繪製訓練歷史圖表"""
    
    # 設定子圖的大小，2行2列的子圖佈局
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 損失函數圖表
    axes[0, 0].plot(history.history['loss'], label='訓練損失', linewidth=2)  # 繪製訓練損失曲線
    if 'val_loss' in history.history:  # 如果驗證損失存在，繪製驗證損失曲線
        axes[0, 0].plot(history.history['val_loss'], label='驗證損失', linewidth=2)
    
    # 設置損失函數圖的標題
    title = '模型損失'
    if fold_num is not None:  # 如果是交叉驗證中的某一折，顯示折數
        title += f' (Fold {fold_num})'
    
    axes[0, 0].set_title(title, fontsize=14, pad=15)  # 設置標題，並加上內邊距
    axes[0, 0].set_xlabel('Epoch', fontsize=12)  # 設置X軸標籤
    axes[0, 0].set_ylabel('Loss', fontsize=12)  # 設置Y軸標籤
    axes[0, 0].legend(fontsize=10)  # 顯示圖例
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)  # 顯示網格，並設置線條樣式

    # 2. 準確率圖表
    axes[0, 1].plot(history.history['accuracy'], label='訓練準確率', linewidth=2)  # 繪製訓練準確率曲線
    if 'val_accuracy' in history.history:  # 如果驗證準確率存在，繪製驗證準確率曲線
        axes[0, 1].plot(history.history['val_accuracy'], label='驗證準確率', linewidth=2)
    
    # 設置準確率圖的標題
    title = '模型準確率'
    if fold_num is not None:  # 如果是交叉驗證中的某一折，顯示折數
        title += f' (Fold {fold_num})'
    
    axes[0, 1].set_title(title, fontsize=14, pad=15)  # 設置標題，並加上內邊距
    axes[0, 1].set_xlabel('Epoch', fontsize=12)  # 設置X軸標籤
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)  # 設置Y軸標籤
    axes[0, 1].legend(fontsize=10)  # 顯示圖例
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)  # 顯示網格，並設置線條樣式

    # 3. AUC圖表 (如果AUC指標存在)
    if 'auc' in history.history and 'val_auc' in history.history:  # 如果AUC和驗證AUC存在
        axes[1, 0].plot(history.history['auc'], label='訓練AUC', linewidth=2)  # 繪製訓練AUC曲線
        axes[1, 0].plot(history.history['val_auc'], label='驗證AUC', linewidth=2)  # 繪製驗證AUC曲線
        
        # 設置AUC圖的標題
        title = '模型AUC'
        if fold_num is not None:  # 如果是交叉驗證中的某一折，顯示折數
            title += f' (Fold {fold_num})'
        
        axes[1, 0].set_title(title, fontsize=14, pad=15)  # 設置標題，並加上內邊距
        axes[1, 0].set_xlabel('Epoch', fontsize=12)  # 設置X軸標籤
        axes[1, 0].set_ylabel('AUC', fontsize=12)  # 設置Y軸標籤
        axes[1, 0].legend(fontsize=10)  # 顯示圖例
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)  # 顯示網格，並設置線條樣式

    # 4. 學習率圖表 (如果學習率歷史存在)
    if 'lr' in history.history:  # 如果學習率歷史存在
        axes[1, 1].plot(history.history['lr'], label='學習率', linewidth=2, color='green')  # 繪製學習率曲線
        
        # 設置學習率圖的標題
        title = '學習率變化'
        if fold_num is not None:  # 如果是交叉驗證中的某一折，顯示折數
            title += f' (Fold {fold_num})'
        
        axes[1, 1].set_title(title, fontsize=14, pad=15)  # 設置標題，並加上內邊距
        axes[1, 1].set_xlabel('Epoch', fontsize=12)  # 設置X軸標籤
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)  # 設置Y軸標籤
        axes[1, 1].legend(fontsize=10)  # 顯示圖例
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)  # 顯示網格，並設置線條樣式
        axes[1, 1].set_yscale('log')  # 設置Y軸為對數尺度，適合學習率這類範圍大幅變動的資料

    # 進行自動調整佈局，避免子圖重疊
    plt.tight_layout()
    
    # 顯示所有圖表
    plt.show()

        


def 訓練模型(X, y, 特徵列, model_path='heart_disease_model.keras'):
    # 初始化评估指标列表
    test_losses = []
    test_accuracies = []
    test_aucs = []
    histories = []
    
    k折 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for 折數, (訓練索引, 驗證索引) in enumerate(k折.split(X, y), 1):
    
        # 在回調中直接使用 model_path

    
    # 使用K折交叉驗證訓練模型，並處理類別不平衡。
    # :param X: 特徵數據
    # :param y: 標籤數據
    # :param 特徵列: 特徵名稱列表
    # """
    # 定義K折交叉驗證
    # 更複雜的交叉驗證策略
   

        # 使用分層抽樣確保每折中類別比例一致
        X_訓練, X_驗證 = X[訓練索引], X[驗證索引]
        y_訓練, y_驗證 = y[訓練索引], y[驗證索引]
        
        # 多重不平衡處理策略

            # 設置類別權重
        smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
        X_訓練, y_訓練 = smote.fit_resample(X_訓練, y_訓練)
        class_weights = {0: 1., 1: 10.}  # 給予少數類別更高權重
        

        
        # 標準化數據
        標準化器 = StandardScaler()
        X_訓練 = 標準化器.fit_transform(X_訓練)
        X_驗證 = 標準化器.transform(X_驗證)
        
        # 建立模型
        模型 = 建立模型(len(特徵列))
        
        # 轉換標籤為類別型態
        y_訓練_類別 = tf.keras.utils.to_categorical(y_訓練)
        y_驗證_類別 = tf.keras.utils.to_categorical(y_驗證)
        
        # 訓練模型
        history = 模型.fit(
            X_訓練, y_訓練_類別, epochs=6000, batch_size=128,
            validation_data=(X_驗證, y_驗證_類別),
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5000, restore_best_weights=True),
            ModelCheckpoint(model_path, save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
        ],
            class_weight=class_weights
        )
        
        # 驗證集預測
        y_預測 = np.argmax(模型.predict(X_驗證), axis=1)
        
        # 評估模型
        test_loss, test_accuracy, test_precision, test_recall, test_auc = 模型.evaluate(X_驗證, y_驗證_類別)
        
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        test_aucs.append(test_auc)
        histories.append(history)
        
        # 繪製混淆矩陣
        繪製混淆矩陣(y_驗證, y_預測, 折數)
        
        # 繪製訓練歷史
        plot_training_history(history, 折數)
        
        print(f"\nFold {折數} Results:")
        print(f"驗證損失: {test_loss:.4f}")
        print(f"驗證準確率: {test_accuracy:.4f}")
        print(f"驗證AUC: {test_auc:.4f}")
    
    # 計算並輸出平均結果
    avg_test_accuracy = np.mean(test_accuracies)
    avg_test_auc = np.mean(test_aucs)
    avg_test_loss = np.mean(test_losses)
    
    print("\nOverall K-Fold Results:")
    print(f"平均測試準確率: {avg_test_accuracy:.4f}")
    print(f"平均測試AUC: {avg_test_auc:.4f}")
    print(f"平均測試損失: {avg_test_loss:.4f}")

if __name__ == "__main__":
    # 定義數據文件路徑
    文件路徑 = r"E:\心臟病3\MLP-Heart-rate-predict\framingham.xlsx"
    
    # 載入並預處理數據
    X, y, 特徵列 = 載入與預處理數據(文件路徑)
    
    # 訓練模型
    訓練模型(X, y, 特徵列)















# 提高模型的準確性可以從以下幾個方面進行調整：

# 數據預處理：

# 特徵標準化：神經網絡模型通常對特徵的範圍和分佈較為敏感，因此對特徵進行標準化或正規化可以提高模型的效果。你可以使用 StandardScaler 或 MinMaxScaler 來標準化特徵數據。
# 處理缺失值：如果數據中有缺失值，應該進行處理。可以選擇丟棄缺失值行，或用均值/中位數等進行填補。
# 模型架構調整：

# 增加層數或神經元：如果模型容量過小，可能無法學到足夠的特徵。你可以嘗試增加神經網絡的層數或每層的神經元數量。
# 改變激活函數：ReLU 是一個很常見的選擇，但有時候可以考慮其他激活函數，如 Leaky ReLU 或 ELU，這些可以避免 ReLU 的「死神經元」問題。
# 使用 Dropout：Dropout 是一種正則化技術，能幫助防止過擬合。你可以在某些層之間加入 Dropout 層。
# 訓練過程調整：

# 增加訓練輪次：2000 輪的訓練可能過多或過少，應該根據損失和準確率的趨勢進行調整。你可以觀察訓練過程中的損失和準確率變化，並進行早停（Early Stopping）來避免過擬合。
# 調整學習率：Adam 優化器的學習率可能需要調整。你可以嘗試不同的學習率，看看哪一個效果最好。
# 批次大小調整：批次大小對訓練速度和結果有影響。你可以試著調整批次大小（例如，32、128）來獲得更好的結果。
# 使用更多的訓練數據：

# 增強數據集：如果數據量不夠多，可以嘗試進行數據增強，例如使用交叉驗證來減少模型的偏差，或者收集更多樣本。
# 合成數據：可以考慮生成合成數據（例如 SMOTE）來平衡樣本數量，尤其是當某些類別的樣本過少時。