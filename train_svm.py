from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.utils.data as Data
import time
import pymysql
import uuid
import joblib

class SVMWorker(QThread):
    progress_update = pyqtSignal(int, str)

    def __init__(self, train_data, test_data, batch_size, C, kernel, class_weight, user_name):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.C = C  # 惩罚系数
        self.kernel = kernel  # 核函数
        self.class_weight = class_weight  # 误差权重
        self.user_name = user_name

    def run(self):
        self.progress_update.emit(0, "Initializing data loader")
        torch.manual_seed(1)

        train_loader = Data.DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.progress_update.emit(10, "Preparing training data")
        train_x = torch.unsqueeze(self.train_data.data, dim=1).type(torch.FloatTensor)[:60000] / 255
        train_y = self.train_data.targets[:60000]
        test_x = torch.unsqueeze(self.test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255
        test_y = self.test_data.targets[:2000]

        # Flatten the images for SVM
        self.progress_update.emit(20, "Flattening images")
        train_x = train_x.view(train_x.size(0), -1).numpy()
        test_x = test_x.view(test_x.size(0), -1).numpy()
        train_y = train_y.numpy()
        test_y = test_y.numpy()

        # Standardize the data
        self.progress_update.emit(30, "Standardizing data")
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

        model_name = 'SVM'  # 模型名称
        experiment_id = str(uuid.uuid4())  # 生成唯一的实验ID
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
        date_time_str = time.strftime("%Y%m%d_%H%M%S")
        
        self.progress_update.emit(50, "Training SVM model")
        # Train the SVM model with provided hyperparameters
        clf = svm.SVC(C=self.C, kernel=self.kernel, class_weight=self.class_weight)
        clf.fit(train_x, train_y)

        self.progress_update.emit(70, "Making predictions")
        # Make predictions
        pred_y = clf.predict(test_x)
        accuracy = accuracy_score(test_y, pred_y)
        
        self.progress_update.emit(80, "Saving model")
        model_filename = f"svm_model_{date_time_str}_{experiment_id}.joblib"
        joblib.dump(clf, model_filename)
        print(f"Model saved as {model_filename}.")
        
        self.progress_update.emit(90, "Saving results to database")
        # Connect to MySQL database and insert results
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='blackxiaobai',
            database='blackxiaobai'
        )

        cursor = connection.cursor()
        insert_query = """
        INSERT INTO result (id, name, accuracy, loss_rate ,date,user_name)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        # Since SVM does not use loss in the same way as neural networks, set loss to None
        cursor.execute(insert_query, (experiment_id, model_name, accuracy, None, start_time,self.user_name))
        connection.commit()
        cursor.close()
        connection.close()
        print("Results saved to database.")
        
        self.progress_update.emit(100, "Completed")

