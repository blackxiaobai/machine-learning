from PyQt5.QtCore import QThread, pyqtSignal
import torch
import torch.nn as nn
import torch.utils.data as Data
import time
import pymysql
import uuid

class CNN(nn.Module):
    def __init__(self, kernel_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size, 1, kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size, 1, kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

class TrainWorker(QThread):
    progress_update = pyqtSignal(int)

    def __init__(self, train_data, test_data, kernel_size, learning_rate, batch_size,user_name, epoch=1):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.user_name = user_name
        self.epoch = epoch

    def run(self):
        torch.manual_seed(1)

        train_loader = Data.DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True
        )

        test_x = torch.unsqueeze(self.test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255
        test_y = self.test_data.targets[:2000]

        cnn = CNN(self.kernel_size)
        optimizer = torch.optim.Adam(cnn.parameters(), lr=self.learning_rate)
        loss_func = nn.CrossEntropyLoss()

        total_steps = len(train_loader) * self.epoch
        current_step = 0

        experiment_id = str(uuid.uuid4())  # 生成唯一的实验ID
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
        model_name = 'CNN'

        for epoch in range(self.epoch):
            for step, (b_x, b_y) in enumerate(train_loader):
                output = cnn(b_x)
                loss = loss_func(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current_step += 1
                progress = int((current_step / total_steps) * 100)
                self.progress_update.emit(progress)

                if step % 50 == 0:
                    test_output = cnn(test_x)
                    pred_y = torch.max(test_output, 1)[1].data.numpy()
                    accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_filename = f'cnn_model_{timestamp}.pkl'
        torch.save(cnn.state_dict(), model_filename)
        print(f"Training completed and model saved as {model_filename}.")

        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',      
            password='blackxiaobai', 
            database='blackxiaobai'
        )

        cursor = connection.cursor()
        insert_query = """
        INSERT INTO result (id, name, Accuracy, loss_rate,date,user_name)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (experiment_id, model_name, accuracy, loss.item(), start_time,self.user_name))
        connection.commit()
        cursor.close()
        connection.close()
        print("Results saved to database.")



