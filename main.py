from login13 import *
from ui23 import *
from mainui21 import *
from PyQt5.QtWidgets import QApplication, QMainWindow,QTableWidget,QTableWidgetItem, QMessageBox,QVBoxLayout, QStackedWidget, QFileDialog
import pymysql
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import pandas as pd
#from sklearn import datasets
from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC
#from sklearn.metrics import accuracy_score, log_loss
#from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import sys
#import os
#import uuid
#from functools import partial
import torch
#import torch.nn as nn
#import torch.optim as optim
import torchvision
#import torchvision.transforms as transforms
#from torchvision.datasets import MNIST
#from torch.utils.data import DataLoader, random_split
#from cryptography.hazmat.primitives import hashes
from PIL import Image
#import torch.nn.functional as F
from train_cnn import TrainWorker,CNN
from train_svm import SVMWorker
from train_rf import RandomForestWorker
import joblib
user_now = ''

class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_LoginWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.pushButton_Login.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(0))
        self.ui.pushButton_Register.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(1))
        self.ui.pushButton.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(2))
        self.ui.pushButton_L_sure.clicked.connect(self.login_in)
        self.ui.pushButton_L_sure_2.clicked.connect(self.login_in1)
        self.ui.pushButton_R_sure.clicked.connect(self.register)
        self.show()

    def login_in(self):
        name = self.ui.lineEdit_R_name_2.text()
        account = self.ui.lineEdit_L_account.text()
        password = self.ui.lineEdit_L_password.text()
        #账号为空的判定
        if not account or not password or not name:
            self.ui.stackedWidget.setCurrentIndex(2)
        #连接数据库
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',      
            password='blackxiaobai', 
            database='blackxiaobai'
        )

        try:
            with connection.cursor() as cursor:
                # Create a new record
                sql = "SELECT * FROM user WHERE account=%s AND password=%s AND name=%s"
                cursor.execute(sql, (account, password,name))
                result = cursor.fetchone()
                if result:
                    self.win = MainWindow(user_name=name)
                    self.close()
                else:
                    self.ui.stackedWidget.setCurrentIndex(3)
        finally:
            connection.close()
        
    def login_in1(self):
        account = self.ui.lineEdit_L_account_2.text()
        password = self.ui.lineEdit_L_password_2.text()
            #账号为空的判定
        if not account or not password:
            self.ui.stackedWidget.setCurrentIndex(2)
        #连接数据库
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',      
            password='blackxiaobai', 
            database='blackxiaobai'
        )

        try:
            with connection.cursor() as cursor:
                # Create a new record
                sql = "SELECT * FROM administrator WHERE account=%s AND password=%s"
                cursor.execute(sql, (account, password))
                result = cursor.fetchone()
                if result:
                    self.close()
                    self.win = MainWindow1(admin_account=account)
                    self.win.show()
                else:
                    self.ui.stackedWidget.setCurrentIndex(3)
        finally:
            connection.close()
    def register(self):
        name = self.ui.lineEdit_R_name.text()
        account = self.ui.lineEdit_R_account_2.text()
        password = self.ui.lineEdit_password_1.text()
        confirm_password = self.ui.lineEdit_password_2.text()
        email = self.ui.lineEdit_R_email.text()
        #账号或密码为空的判定
        if not account or not password or not confirm_password:
            self.ui.stackedWidget.setCurrentIndex(2)
            return
        #确认密码不一致的判定
        if password != confirm_password:
            self.ui.stackedWidget.setCurrentIndex(4)
            return
        #连接数据库
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',      
            password='blackxiaobai', 
            database='blackxiaobai'
        )

        try:
            with connection.cursor() as cursor:
                sql_check = "SELECT * FROM user WHERE name = %s"
                cursor.execute(sql_check, (name,))
                result = cursor.fetchone()
                if result:
                    self.ui.stackedWidget.setCurrentIndex(5)
                    return
                sql = "INSERT INTO user (account, password, name, Email) VALUES (%s, %s, %s, %s)"
                cursor.execute(sql, (account, password, name, email))
                connection.commit()
                self.ui.stackedWidget.setCurrentIndex(1)  # Registration successful page
        finally:
            connection.close()

class PredictionThread(QThread):
    prediction_done = pyqtSignal(str)

    def __init__(self, model, test_images):
        super(PredictionThread, self).__init__()
        self.model = model
        self.test_images = test_images

    def run(self):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # 假设 test_images 是一个 numpy 数组
            test_tensor = torch.tensor(self.test_images, dtype=torch.float32)
            # 将输入数据的形状调整为 [batch_size, channels, height, width]
            if len(test_tensor.shape) == 2 and test_tensor.shape[1] == 784:
                test_tensor = test_tensor.view(-1, 1, 28, 28)
            elif len(test_tensor.shape) == 3 and test_tensor.shape[1] == 28 and test_tensor.shape[2] == 28:
                test_tensor = test_tensor.unsqueeze(0)  # Add batch dimension if missing
            output = self.model(test_tensor)
            predicted = torch.argmax(output, 1).item()
            self.prediction_done.emit(f"Predicted Class: {predicted}")

class PredictionThread1(QThread):
    prediction_done = pyqtSignal(str)

    def __init__(self, model, test_images):
        super().__init__()
        self.model = model
        self.test_images = test_images

    def run(self):
        try:
            prediction = self.model.predict(self.test_images)
            self.prediction_done.emit(f"Prediction: {prediction[0]}")
        except Exception as e:
            self.prediction_done.emit(f"Prediction failed: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self,user_name):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.user_name = user_name
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.pushButton_2.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.pushButton_5.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(5))
        self.ui.pushButton_6.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(2))
        self.ui.pushButton_7.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(3))
        self.ui.pushButton_8.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(4))
        self.ui.pushButton_9.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.pushButton_model_1.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(0))
        self.ui.pushButton_model_2.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(1))
        self.ui.pushButton_model_3.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(2))
        self.ui.pushButton_logout.clicked.connect(self.log_out)
        self.ui.pushButton_select_data.clicked.connect(self.open_folder_dialog)
        self.ui.lineEdit_parameter_7.setText('5')
        self.ui.lineEdit_parameter_8.setText('0.001')
        self.ui.lineEdit_parameter_9.setText('50')
        self.ui.pushButton_sure_4.clicked.connect(self.train_cnn_model)
        self.ui.lineEdit_parameter_1.setText('1.0')
        self.ui.lineEdit_parameter_2.setText('linear')
        self.ui.lineEdit_parameter_3.setText('None')
        self.ui.pushButton_sure_2.clicked.connect(self.train_svm_model)
        self.ui.lineEdit_parameter_4.setText('10')
        self.ui.lineEdit_parameter_5.setText('0.5')
        self.ui.lineEdit_parameter_6.setText('20')
        #self.ui.lineEdit_rf_parameter_4.setText('2')  # 批量大小
        self.ui.pushButton_sure_3.clicked.connect(self.train_rf_model)
        self.ui.pushButton_select_model_2.clicked.connect(self.select_model)
        self.ui.pushButton_select_model_3.clicked.connect(self.select_model1)
        self.ui.pushButton_choose_3.clicked.connect(self.select_test_data)
        self.ui.pushButton_start.clicked.connect(self.predict)
        self.ui.pushButton_start_1.clicked.connect(self.predict1)
        self.ui.pushButton_7.clicked.connect(self.fetch_user_experiment_results)
        self.ui.pushButton_save_user_info.clicked.connect(self.save_admin_info)
        self.show()
        
        self.fetch_user_info()
    def fetch_user_experiment_results(self):
        # 连接数据库并获取当前用户的实验数据
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='blackxiaobai',
            database='blackxiaobai'
        )

        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM result WHERE user_name = %s", (self.user_name,))
                experiments = cursor.fetchall()
                self.display_user_experiments(experiments)
                
        finally:
            connection.close()

    def display_user_experiments(self, experiments):
        # 在QTableWidget中展示实验数据
        self.ui.tableWidget.setRowCount(len(experiments))
        self.ui.tableWidget.setColumnCount(6)
        self.ui.tableWidget.setHorizontalHeaderLabels(['ID', 'Name', 'Accuracy', 'Loss Rate', 'Date', 'User Name'])
        for row_idx, experiment in enumerate(experiments):
            self.ui.tableWidget.setItem(row_idx, 0, QTableWidgetItem(str(experiment[0])))
            self.ui.tableWidget.setItem(row_idx, 1, QTableWidgetItem(str(experiment[1])))
            self.ui.tableWidget.setItem(row_idx, 2, QTableWidgetItem(str(experiment[2])))
            self.ui.tableWidget.setItem(row_idx, 3, QTableWidgetItem(str(experiment[3])))
            self.ui.tableWidget.setItem(row_idx, 4, QTableWidgetItem(str(experiment[4])))
            self.ui.tableWidget.setItem(row_idx, 5, QTableWidgetItem(str(experiment[5])))
    


    def select_model1(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.joblib);;All Files (*)", options=options)
        if file_name:
            #self.model_label.setText(f"Model selected")
            try:
                with open(file_name, 'rb') as file:
                    self.model = pickle.load(file)
            except pickle.UnpicklingError:
                try:
                    self.model = joblib.load(file_name)
                except Exception as e:
                    self.show_error_message(f"Failed to load model: {str(e)}")
            except Exception as e:
                self.show_error_message(f"Failed to load model: {str(e)}")

    def select_model(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pkl);;All Files (*)", options=options)
        if file_name:
            #self.model_label.setText("Model selected")
            self.show_message("模型选择成功！")
            try:
                device = torch.device('cpu')
                self.model = CNN(kernel_size=5)  # Ensure kernel_size matches the model
                self.model.load_state_dict(torch.load(file_name, map_location=device))
            except Exception as e:
                self.show_error_message(f"Failed to load model: {str(e)}")

    def select_test_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Test Data", "", "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            try:
                image = Image.open(file_name)
                image = image.convert('L')  # Convert to grayscale
                image = image.resize((28, 28))  # Resize to 28x28
                self.test_images = np.array(image).reshape(1, -1) / 255.0  # Flatten and normalize

                # Display the selected image in the label
                pixmap = QPixmap(file_name)
                self.ui.label_4.setPixmap(pixmap)
                self.ui.label_4.setScaledContents(True)
            except Exception as e:
                self.show_error_message(f"Failed to load test data:str(e)")

    def predict(self):
        if self.model is None or self.test_images is None:
            self.show_error_message("Please select both a model and test data")
            return
        
        self.prediction_thread = PredictionThread(self.model, self.test_images)
        self.prediction_thread.prediction_done.connect(self.show_prediction_result)
        self.prediction_thread.start()

    def predict1(self):
        if self.model is None or self.test_images is None:
            self.show_error_message("Please select both a model and test data")
            return

        self.prediction_thread = PredictionThread1(self.model, self.test_images)
        self.prediction_thread.prediction_done.connect(self.show_prediction_result)
        self.prediction_thread.start()

    def show_prediction_result(self, result):
        self.ui.label_14.setText(result)
    def log_out(self):
        self.close()
        self.login = LoginWindow()
        user_now = ''

    def open_folder_dialog(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if folder:
            #self.label.setText(f"Selected folder: {folder}")
            self.load_data(folder)
    
    def load_data(self, folder_path):
        self.train_data = torchvision.datasets.MNIST(
            root=folder_path,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=False,
        )
        self.test_data = torchvision.datasets.MNIST(
            root=folder_path,
            train=False
        )
        self.show_message("Data loaded successfully.")
    
    def show_message(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("Information")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(message)
        msg_box.setWindowTitle("Error")
        msg_box.exec_()

    def update_progress3(self, value):
        self.ui.progressBar_3.setValue(value)

    def update_progress1(self, progress, message):
        QMessageBox.information(self, "Progress", f"{message}: {progress}%")
    
    def update_progress2(self, value):
        # 使用QMessageBox显示进度
        QMessageBox.information(self, "Progress Update", f"Training Progress: {value}%")

    def train_cnn_model(self):
        if self.train_data is None or self.test_data is None:
            self.show_message("Please select MNIST data first.")
            return

        kernel_size = int(self.ui.lineEdit_parameter_7.text())
        learning_rate = float(self.ui.lineEdit_parameter_8.text())
        batch_size = int(self.ui.lineEdit_parameter_9.text())

        self.worker = TrainWorker(self.train_data, self.test_data, kernel_size, learning_rate, batch_size,self.user_name)
        self.worker.progress_update.connect(self.update_progress3)
        self.worker.finished.connect(lambda: self.show_message("Training completed and model saved."))
        self.worker.start()

    def train_svm_model(self):
        print("SVM training started.")
        if self.train_data is None or self.test_data is None:
            self.show_message("Please select MNIST data first.")
            return

        C = float(self.ui.lineEdit_parameter_1.text())
        kernel = self.ui.lineEdit_parameter_2.text()
        class_weight = self.ui.lineEdit_parameter_3.text()

        # 将 class_weight 转换为 None，如果它的值是 'None'
        if class_weight.lower() == 'none':
            class_weight = None

        #batch_size = int(self.ui.lineEdit_parameter_batch_size.text())
        batch_size = 2
        self.svm_worker = SVMWorker(self.train_data, self.test_data, batch_size, C, kernel, class_weight,self.user_name)
        self.svm_worker.progress_update.connect(self.update_progress1)
        self.show_message("模型训练中，请稍等...")
        self.svm_worker.finished.connect(lambda: self.show_message("SVM Training completed and model saved."))
        self.svm_worker.start()
    
    def train_rf_model(self):
        if self.train_data is None or self.test_data is None:
            self.show_message("Please select MNIST data first.")
            return

        try:
            n_estimators = int(self.ui.lineEdit_parameter_4.text())
        
            max_features = self.ui.lineEdit_parameter_5.text()
            if max_features.isdigit():
                max_features = int(max_features)
            elif max_features.replace('.', '', 1).isdigit():
                max_features = float(max_features)
                if not (0.0 < max_features <= 1.0):
                    self.show_message("Invalid value for max_features. Float values must be in the range (0.0, 1.0].")
                    return
            elif max_features in ['sqrt', 'log2', 'None']:
                max_features = None if max_features == 'None' else max_features
            else:
                self.show_message("Invalid value for max_features. It must be an integer, float in (0.0, 1.0], 'sqrt', 'log2', or 'None'.")
                return
        
            max_depth = self.ui.lineEdit_parameter_6.text()
            max_depth = None if max_depth == 'None' else int(max_depth)
        
            batch_size = 2

            print(f"Parameters: n_estimators={n_estimators}, max_features={max_features}, max_depth={max_depth}, batch_size={batch_size}")  # 调试输出

            self.rf_worker = RandomForestWorker(self.train_data, self.test_data, batch_size, n_estimators, max_features, max_depth, self.user_name)
            self.show_message("模型训练中，请稍等...")
            self.rf_worker.progress_update.connect(self.update_progress2)
            self.rf_worker.finished.connect(lambda: self.show_message("Random Forest Training completed and model saved."))
            self.rf_worker.start()

        except ValueError as e:
            self.show_message(f"Error: Invalid input value. {str(e)}")
        except Exception as e:
            self.show_message(f"Error: {str(e)}")

    def update_progress(self, value):
        # 更新进度的占位函数
        print(f"Progress: {value}%")
    

    def fetch_user_info(self):
        # Connect to the database and get the admin info
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='blackxiaobai',
            database='blackxiaobai'
        )
        
        try:
            with connection.cursor() as cursor:
                sql = "SELECT account,password,Email FROM user WHERE name = %s"
                cursor.execute(sql, (self.user_name,))
                user_info = cursor.fetchone()
                if user_info:
                    self.ui.lineEdit_user_account_2.setText(user_info[0])
                    self.ui.lineEdit_user_password.setText(user_info[1])
                    self.ui.lineEdit_user_email.setText(user_info[2])
                else:
                    QMessageBox.warning(self, "Error", "Admin account not found.")
        finally:
            connection.close()

    def save_admin_info(self):
        # Get the updated admin info from the UI
        new_account = self.ui.lineEdit_user_account_2.text()
        new_password = self.ui.lineEdit_user_password.text()
        new_Email = self.ui.lineEdit_user_email.text()
        
        # Connect to the database and update the admin info
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='blackxiaobai',
            database='blackxiaobai'
        )
        
        try:
            with connection.cursor() as cursor:
                sql = "UPDATE user SET account = %s,password = %s, Email = %s WHERE account = %s"
                cursor.execute(sql, (new_account,new_password,new_Email, self.user_name))
                connection.commit()
                QMessageBox.information(self, "Success", "Admin info updated successfully.")
                #self.admin_account = new_account
        finally:
            connection.close()
    

class MainWindow1(QMainWindow):
    def __init__(self,admin_account):
        super().__init__()
        self.admin_account = admin_account
        self.ui = Ui_MainWindow1()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.pushButton_2.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.pushButton_5.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(2))
        self.ui.pushButton_6.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(3))
        self.ui.pushButton_7.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.pushButton_logout.clicked.connect(self.log_out)
        self.ui.pushButton_search_user.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(1))
        self.ui.pushButton_search.clicked.connect(self.search_user_by_name)
        self.ui.pushButton_fetch_users.clicked.connect(self.fetch_users)
        self.ui.pushButton_add_user.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(2))
        self.ui.pushButton_update.clicked.connect(self.update_user_by_name)
        self.ui.pushButton_delete_user.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(3))
        self.ui.pushButton_delete.clicked.connect(self.delete_user_by_name)
        self.ui.pushButton_view_all_experiments.clicked.connect(self.fetch_experiment_results)
        self.ui.pushButton_8.clicked.connect(lambda: self.ui.stackedWidget_3.setCurrentIndex(1))
        self.ui.pushButton_search_experiment.clicked.connect(self.search_experiment_by_id)
        self.ui.pushButton_9.clicked.connect(lambda: self.ui.stackedWidget_3.setCurrentIndex(2))
        self.ui.pushButton_update_2.clicked.connect(self.update_experiment_by_id)
        self.ui.pushButton_delete_experiment_2.clicked.connect(lambda: self.ui.stackedWidget_3.setCurrentIndex(3))
        self.ui.pushButton_delete_2.clicked.connect(self.delete_experiment_by_id)
        self.ui.pushButton_save_admin_info.clicked.connect(self.save_admin_info)

        self.show()
        self.fetch_admin_info()

    def log_out(self):
        self.close()
        self.login = LoginWindow()
        user_now = ''

    def fetch_users(self):
        # 连接数据库并获取所有用户信息
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='blackxiaobai',
            database='blackxiaobai'
        )

        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM user")
                users = cursor.fetchall()
                self.ui.stackedWidget_2.setCurrentIndex(0)
                self.display_users(users)
                
        finally:
            connection.close()
    
    def display_users(self, users):
        # 在QTableWidget中展示用户信息
        self.ui.tableWidget.setRowCount(len(users))
        self.ui.tableWidget.setColumnCount(4)  
        self.ui.tableWidget.setHorizontalHeaderLabels(['Name','Account', 'Password','Email'])
        for row_idx, user in enumerate(users):
            self.ui.tableWidget.setItem(row_idx, 0, QTableWidgetItem(user[0]))
            self.ui.tableWidget.setItem(row_idx, 1, QTableWidgetItem(user[1]))
            self.ui.tableWidget.setItem(row_idx, 2, QTableWidgetItem(user[2]))
            self.ui.tableWidget.setItem(row_idx, 3, QTableWidgetItem(user[3]))
    def display_users1(self, users):
        # 在QTableWidget中展示用户信息
        self.ui.tableWidget_4.setRowCount(len(users))
        self.ui.tableWidget_4.setColumnCount(4)  
        self.ui.tableWidget_4.setHorizontalHeaderLabels(['Name', 'Account', 'Password','Email'])
        for row_idx, user in enumerate(users):
            self.ui.tableWidget_4.setItem(row_idx, 0, QTableWidgetItem(user[0]))
            self.ui.tableWidget_4.setItem(row_idx, 1, QTableWidgetItem(user[1]))
            self.ui.tableWidget_4.setItem(row_idx, 2, QTableWidgetItem(user[2]))
            self.ui.tableWidget_4.setItem(row_idx, 3, QTableWidgetItem(user[3]))
    def search_user_by_name(self):
        name = self.ui.lineEdit_searchName.text()
        if not name:
            QMessageBox.warning(self, "Input Error", "Name cannot be empty")
            return
        
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='blackxiaobai',
            database='blackxiaobai'
        )

        try:
            with connection.cursor() as cursor:
                sql = "SELECT name, account, password, Email FROM user WHERE name = %s"
                cursor.execute(sql, (name,))
                users = cursor.fetchall()
                if users:
                    self.display_users1(users)
                    self.ui.stackedWidget_2.setCurrentIndex(1)
                else:
                    QMessageBox.information(self, "Search Result", "No user found with the given name")
        finally:
            connection.close()
    
    def update_user_by_name(self):
        name = self.ui.lineEdit_updateName.text()
        new_account = self.ui.lineEdit_updateAccount.text()
        new_password = self.ui.lineEdit_updatePassword.text()
        new_email = self.ui.lineEdit_update_email.text()
        if not name or not new_account or not new_password or not new_email:
            QMessageBox.warning(self, "Input Error", "Name, account, and password cannot be empty")
            return
        
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='blackxiaobai',
            database='blackxiaobai'
        )

        try:
            with connection.cursor() as cursor:
                sql = "UPDATE user SET account = %s, password = %s,Email = %s WHERE name = %s"
                cursor.execute(sql, (new_account, new_password,new_email,name))
                connection.commit()
                if cursor.rowcount > 0:
                    QMessageBox.information(self, "Success", "User information updated successfully")
                else:
                    QMessageBox.information(self, "Update Failed", "No user found with the given name")
        finally:
            connection.close()

    def delete_user_by_name(self):
        name = self.ui.lineEdit_delete_name.text()
        if not name:
            QMessageBox.warning(self, "Delete Error", "Please enter a name.")
            return

        # Connect to the database and delete the user by name
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='blackxiaobai',
            database='blackxiaobai'
        )

        try:
            with connection.cursor() as cursor:
                # Check if the user exists
                sql_check = "SELECT * FROM user WHERE name = %s"
                cursor.execute(sql_check, (name,))
                user = cursor.fetchone()
                if not user:
                    QMessageBox.warning(self, "Delete Error", "User not found.")
                    return
                
                # Delete the user
                sql_delete = "DELETE FROM user WHERE name = %s"
                cursor.execute(sql_delete, (name,))
                connection.commit()
                QMessageBox.information(self, "Success", "User deleted successfully.")
                self.fetch_users()  # Refresh the user list
        finally:
            connection.close()
    
    def fetch_experiment_results(self):
        # 连接数据库并获取所有实验数据
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='blackxiaobai',
            database='blackxiaobai'
        )

        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM result")
                experiments = cursor.fetchall()
                self.ui.stackedWidget_3.setCurrentIndex(0)
                self.display_experiments(experiments)
                
        finally:
            connection.close()

    def display_experiments(self, experiments):
        # 在QTableWidget中展示实验数据
        self.ui.tableWidget_3.setRowCount(len(experiments))
        self.ui.tableWidget_3.setColumnCount(6)
        self.ui.tableWidget_3.setHorizontalHeaderLabels(['ID',  'Name', 'Accuracy', 'Loss Rate','date','user_name'])
        for row_idx, experiment in enumerate(experiments):
            self.ui.tableWidget_3.setItem(row_idx, 0, QTableWidgetItem(str(experiment[0])))
            self.ui.tableWidget_3.setItem(row_idx, 1, QTableWidgetItem(str(experiment[1])))
            self.ui.tableWidget_3.setItem(row_idx, 2, QTableWidgetItem(str(experiment[2])))
            self.ui.tableWidget_3.setItem(row_idx, 3, QTableWidgetItem(str(experiment[3])))
            self.ui.tableWidget_3.setItem(row_idx, 4, QTableWidgetItem(str(experiment[4])))
            self.ui.tableWidget_3.setItem(row_idx, 5, QTableWidgetItem(str(experiment[5])))
            
    def display_experiments1(self, experiments):
        # 在QTableWidget中展示实验数据
        self.ui.tableWidget_2.setRowCount(len(experiments))
        self.ui.tableWidget_2.setColumnCount(6)
        self.ui.tableWidget_2.setHorizontalHeaderLabels(['ID', 'Name', 'Accuracy', 'Loss Rate','date','user_name'])
        for row_idx, experiment in enumerate(experiments):
            self.ui.tableWidget_2.setItem(row_idx, 0, QTableWidgetItem(str(experiment[0])))
            self.ui.tableWidget_2.setItem(row_idx, 1, QTableWidgetItem(str(experiment[1])))
            self.ui.tableWidget_2.setItem(row_idx, 2, QTableWidgetItem(str(experiment[2])))
            self.ui.tableWidget_2.setItem(row_idx, 3, QTableWidgetItem(str(experiment[3])))
            self.ui.tableWidget_2.setItem(row_idx, 4, QTableWidgetItem(str(experiment[4])))
            self.ui.tableWidget_2.setItem(row_idx, 5, QTableWidgetItem(str(experiment[5])))

    def search_experiment_by_id(self):
        experiment_id = self.ui.lineEdit_searchExperimentID.text()
        if not experiment_id:
            QMessageBox.warning(self, "Input Error", "Experiment ID cannot be empty")
            return
        
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='blackxiaobai',
            database='blackxiaobai'
        )
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM result WHERE id = %s"
                cursor.execute(sql, (experiment_id,))
                experiments = cursor.fetchall()
                if experiments:
                    self.display_experiments1(experiments)
                    self.ui.stackedWidget_3.setCurrentIndex(1)
                else:
                    QMessageBox.information(self, "Search Result", "No experiment found with the given ID")
        finally:
            connection.close()

    def update_experiment_by_id(self):
        ID_experiment = self.ui.lineEdit_updateName_2.text()
        new_Name = self.ui.lineEdit_updateName_4.text()
        new_Accuracy = self.ui.lineEdit_updateName_5.text()
        new_Loss_rate = self.ui.lineEdit_updateName_6.text()

        if not ID_experiment  or not new_Name or not new_Accuracy or not new_Loss_rate:
            QMessageBox.warning(self, "Input Error", "Anything cannot be empty")
            return
        
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='blackxiaobai',
            database='blackxiaobai'
        )

        try:
            with connection.cursor() as cursor:
                sql = "UPDATE result SET name = %s, Accuracy = %s, loss_rate = %s WHERE id = %s"
                cursor.execute(sql, (new_Name,new_Accuracy,new_Loss_rate,ID_experiment))
                connection.commit()
                if cursor.rowcount > 0:
                    QMessageBox.information(self, "Success", "Experiment information updated successfully")
                else:
                    QMessageBox.information(self, "Update Failed", "No result found with the given ID")
        finally:
            connection.close()

    def delete_experiment_by_id(self):
        experiment_id = self.ui.lineEdit_delete_name_2.text()
        if not experiment_id:
            QMessageBox.warning(self, "Delete Error", "Please enter an experiment ID.")
            return

    # Connect to the database and delete the experiment by id
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='blackxiaobai',
            database='blackxiaobai'
        )

        try:
            with connection.cursor() as cursor:
            # Check if the experiment exists
                sql_check = "SELECT * FROM result WHERE id = %s"
                cursor.execute(sql_check, (experiment_id,))
                experiment = cursor.fetchone()
                if not experiment:
                    QMessageBox.warning(self, "Delete Error", "Experiment ID not found.")
                    return
            
            # Delete the experiment
                sql_delete = "DELETE FROM result WHERE id = %s"
                cursor.execute(sql_delete, (experiment_id,))
                connection.commit()
                QMessageBox.information(self, "Success", "Experiment deleted successfully.")
                self.fetch_experiment_results()  # Update the table with current experiments
        finally:
            connection.close()

    def fetch_admin_info(self):
        # Connect to the database and get the admin info
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='blackxiaobai',
            database='blackxiaobai'
        )
        
        try:
            with connection.cursor() as cursor:
                sql = "SELECT account, password FROM administrator WHERE account = %s"
                cursor.execute(sql, (self.admin_account,))
                admin_info = cursor.fetchone()
                if admin_info:
                    self.ui.lineEdit_admin_account.setText(admin_info[0])
                    self.ui.lineEdit_admin_password.setText(admin_info[1])
                else:
                    QMessageBox.warning(self, "Error", "Admin account not found.")
        finally:
            connection.close()

    def save_admin_info(self):
        # Get the updated admin info from the UI
        new_account = self.ui.lineEdit_admin_account.text()
        new_password = self.ui.lineEdit_admin_password.text()
        
        # Connect to the database and update the admin info
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='blackxiaobai',
            database='blackxiaobai'
        )
        
        try:
            with connection.cursor() as cursor:
                sql = "UPDATE administrator SET account = %s, password = %s WHERE account = %s"
                cursor.execute(sql, (new_account, new_password, self.admin_account))
                connection.commit()
                QMessageBox.information(self, "Success", "Admin info updated successfully.")
                self.admin_account = new_account
        finally:
            connection.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = LoginWindow()
    sys.exit(app.exec_()) 