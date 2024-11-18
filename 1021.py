from sklearn.preprocessing import StandardScaler  # 標準化數據
from sklearn.model_selection import KFold  # K折交叉驗證
from imblearn.over_sampling import SMOTE  # 用於過取樣的SMOTE方法
from imblearn.under_sampling import RandomUnderSampler  # 用於欠取樣的隨機欠取樣方法
from imblearn.pipeline import Pipeline  # 用於將過取樣和欠取樣組合的管道
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # 用於訓練過程中的回調
from sklearn.metrics import classification_report, confusion_matrix  # 用於生成分類報告和混淆矩陣
import joblib  # 用於保存模型和標準化器
import matplotlib.pyplot as plt  # 用於繪圖
import seaborn as sns  # 用於繪製混淆矩陣
import tensorflow as tf  # 用於深度學習
import numpy as np  # 用於數據處理
import pandas as pd  # 用於數據處理
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 設定中文字型，這裡以 SimHei 為例
plt.rcParams['font.family'] = 'SimHei'  # 使用 SimHei 字型

# 如果沒有安裝 SimHei，可以使用以下語句指定系統中的字型
# plt.rcParams['font.family'] = font_manager.FontProperties(fname='path_to_your_font').get_name()

# 設定隨機種子，確保結果可重現
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(file_path, balance_strategy='combine'):
    """載入並預處理數據，包含數據平衡"""
    df = pd.read_excel(file_path)  # 從Excel文件中載入數據

    # 定義需要的特徵列
    features = ['age', 'male', 'education', 'currentSmoker', 'cigsPerDay', 
                'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 
                'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    
    # 用平均值填補缺失值
    df[features] = df[features].fillna(df[features].mean())
    
    X = df[features].values.astype(np.float64)  # 提取特徵
    y = df["TenYearCHD"].values.astype(np.int8)  # 提取目標變數

    return X, y, features  # 返回特徵、目標變數和特徵名

def create_model(input_dim, category=2):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_dim=input_dim),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(category, activation='softmax')
    ])
    
    # 定義焦點損失函數
    def focal_loss(gamma=2., alpha=.25):
        def focal_loss_fixed(y_true, y_pred):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -tf.reduce_mean(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1 + 1e-7)) - \
                   tf.reduce_mean((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0 + 1e-7))
        return focal_loss_fixed
    
    # 編譯模型，使用Adam優化器、焦點損失函數和AUC評價指標
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_loss(gamma=2., alpha=.25),
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model
def plot_confusion_matrix(y_true, y_pred, fold_num=None):
    """繪製混淆矩陣"""
    # 設定分類名稱
    classes = ['無心臟病', '有心臟病']
    
    # 計算混淆矩陣
    cm = confusion_matrix(y_true, y_pred)
    
    # 設定圖的大小，這裡縮小為 (6, 5)
    plt.figure(figsize=(6, 5))
    
    # 繪製熱圖，使用藍色調
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    # 標題設定
    title = '混淆矩陣'
    if fold_num is not None:
        title += f' (Fold {fold_num})'
    plt.title(title, fontsize=14)
    
    # X、Y軸標籤
    plt.xlabel('預測標籤', fontsize=12)
    plt.ylabel('真實標籤', fontsize=12)
    
    # 顯示圖表
    plt.show()

def plot_training_history(history, fold_num=None):
    """繪製訓練歷史圖表"""
    # 設定子圖大小，這裡縮小為 (12, 10)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 損失函數圖
    axes[0, 0].plot(history.history['loss'], label='訓練損失', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='驗證損失', linewidth=2)
    title = '模型損失'
    if fold_num is not None:
        title += f' (Fold {fold_num})'
    axes[0, 0].set_title(title, fontsize=14, pad=15)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # 準確率圖
    axes[0, 1].plot(history.history['accuracy'], label='訓練準確率', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='驗證準確率', linewidth=2)
    title = '模型準確率'
    if fold_num is not None:
        title += f' (Fold {fold_num})'
    axes[0, 1].set_title(title, fontsize=14, pad=15)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # AUC 圖
    if 'auc' in history.history:
        axes[1, 0].plot(history.history['auc'], label='訓練AUC', linewidth=2)
        axes[1, 0].plot(history.history['val_auc'], label='驗證AUC', linewidth=2)
        title = '模型AUC'
        if fold_num is not None:
            title += f' (Fold {fold_num})'
        axes[1, 0].set_title(title, fontsize=14, pad=15)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('AUC', fontsize=12)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # 學習率圖
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='學習率', linewidth=2, color='green')
        title = '學習率變化'
        if fold_num is not None:
            title += f' (Fold {fold_num})'
        axes[1, 1].set_title(title, fontsize=14, pad=15)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        axes[1, 1].set_yscale('log')
        
                # 進行自動調整佈局
        plt.tight_layout()
        plt.show()

def train_model(X, y, features, model_path='heart_disease_model.h5', 
                scaler_path='scaler.pkl', balance_strategy='combine'):
    """訓練模型的主要函數，包含K-Fold交叉驗證"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5折交叉驗證
    history_list = []  # 保存每折的訓練歷史
    test_accuracies = []  # 保存每折的測試準確率
    test_aucs = []  # 保存每折的測試AUC
    test_losses = []  # 保存每折的測試損失
    
    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        print(f"\nTraining Fold {fold}")
        
        X_train, X_val = X[train_index], X[val_index]  # 切分訓練集和驗證集
        y_train, y_val = y[train_index], y[val_index]
        
        # 資料平衡處理
        if balance_strategy == 'smote':  # 使用SMOTE過取樣
            sampler = SMOTE(random_state=42)
            X_train, y_train = sampler.fit_resample(X_train, y_train)
        elif balance_strategy == 'combine':  # 使用SMOTE和隨機欠取樣的結合方法
            over = SMOTE(sampling_strategy=0.6, random_state=42)
            under = RandomUnderSampler(sampling_strategy=0.7, random_state=42)
            steps = [('over', over), ('under', under)]
            pipeline = Pipeline(steps=steps)
            X_train, y_train = pipeline.fit_resample(X_train, y_train)
        
        # 標準化數據
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 轉換為 one-hot 編碼
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
        y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=2)
        
        # 創建並訓練模型
        model = create_model(len(features))
        
        # 設置回調函數
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=0.00001, verbose=1),
            ModelCheckpoint(f'model_fold_{fold}.keras', monitor='val_loss', save_best_only=True, verbose=1)
        ]
        
        # 訓練模型
        history = model.fit(
            X_train_scaled, y_train_cat,
            epochs=500,
            batch_size=32,
            validation_data=(X_val_scaled, y_val_cat),
            class_weight={0: 1, 1: 15},
            callbacks=callbacks,
            verbose=1
        )
        
        # 儲存歷史數據
        history_list.append(history)
        
        # 評估模型
        test_loss, test_accuracy, test_auc = model.evaluate(X_val_scaled, y_val_cat)
        test_accuracies.append(test_accuracy)
        test_aucs.append(test_auc)
        test_losses.append(test_loss)
        
        # 預測並繪製混淆矩陣
        y_pred = model.predict(X_val_scaled)
        y_pred_classes = np.argmax(y_pred, axis=1)
        plot_confusion_matrix(y_val, y_pred_classes, fold)
            # 生成分類報告
        report = classification_report(y_val, y_pred_classes, target_names=['無心臟病', '有心臟病'])
        print("\n分類報告:\n", report)   
        # 繪製訓練歷史
        plot_training_history(history, fold)
        
        print(f"\nFold {fold} Results:")
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
    
    return history_list, avg_test_accuracy, avg_test_auc, avg_test_loss

if __name__ == "__main__":
    # 載入並平衡數據
    X, y, features = load_and_preprocess_data('framingham.xlsx', balance_strategy='combine')
    
    # 訓練模型並顯示結果
    history_list, avg_test_accuracy, avg_test_auc, avg_test_loss = train_model(X, y, features)













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