from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.utils.data as Data
import time
import pymysql
import uuid
import joblib  # 用于保存模型

class RandomForestWorker(QThread):
    progress_update = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, train_data, test_data, batch_size, n_estimators, max_features, max_depth,user_name):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.user_name = user_name
        self.batch_size = batch_size
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth

    def run(self):
        torch.manual_seed(1)

        train_loader = Data.DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True
        )

        train_x = torch.unsqueeze(self.train_data.data, dim=1).type(torch.FloatTensor)[:60000] / 255
        train_y = self.train_data.targets[:60000]
        test_x = torch.unsqueeze(self.test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255
        test_y = self.test_data.targets[:2000]

        # Flatten the images for RandomForest
        train_x = train_x.view(train_x.size(0), -1).numpy()
        test_x = test_x.view(test_x.size(0), -1).numpy()
        train_y = train_y.numpy()
        test_y = test_y.numpy()

        # Standardize the data
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

        model_name = 'RandomForest'  # 模型名称
        experiment_id = str(uuid.uuid4())  # 生成唯一的实验ID
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
        date_time_str = time.strftime("%Y%m%d_%H%M%S")  # 获取用于文件名的日期时间字符串

        # Train the RandomForest model with provided hyperparameters
        clf = RandomForestClassifier(n_estimators=self.n_estimators, max_features=self.max_features, max_depth=self.max_depth)

        for i in range(self.n_estimators):
            clf.fit(train_x, train_y)
            progress = int((i / self.n_estimators) * 100)
            self.progress_update.emit(progress)
            time.sleep(0.1)  # 模拟训练时间

        # 发出100%的进度信号
        self.progress_update.emit(100)

        # Make predictions
        pred_y = clf.predict(test_x)
        accuracy = accuracy_score(test_y, pred_y)

        print(f"Training completed with accuracy: {accuracy:.2f}")
        
        model_filename = f"rf_model_{date_time_str}_{experiment_id}.joblib"
        joblib.dump(clf, model_filename)
        print(f"Model saved as {model_filename}.")

        # Connect to MySQL database and insert results
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='blackxiaobai',
            database='blackxiaobai'
        )

        cursor = connection.cursor()
        insert_query = """
        INSERT INTO result (id, name, accuracy, loss_rate, date,user_name)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        # Since RandomForest does not use loss in the same way as neural networks, set loss to None
        cursor.execute(insert_query, (experiment_id, 'RandomForest Experiment', accuracy, None, start_time, self.user_name))
        connection.commit()
        cursor.close()
        connection.close()
        print("Results saved to database.")

        # 发出完成信号
        self.finished.emit()
