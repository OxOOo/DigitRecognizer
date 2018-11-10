# 《人工智能导论》实验三

## 实验结果

<div align=center>
    <table>
        <thead>
            <tr>
                <th>算法</th>
                <th>正确率</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>CNN</td><td>0.99086</td><td>训练了一晚上</td>
            </tr>
            <tr>
                <td>CNN(small)</td><td>0.98686</td><td>训练时间较短</td>
            </tr>
            <tr>
                <td>KNN(K=1)</td><td>0.97114</td><td>跑得贼块</td>
            </tr>
            <tr>
                <td>KNN(K=5)</td><td>0.96800</td><td>跑得贼块</td>
            </tr>
            <tr>
                <td>KNN(K=3)</td><td>0.96857</td><td>跑得贼块</td>
            </tr>
        </tbody>
    </table>
</div>

## 文件说明

* cnn.csv: 用CNN模型跑出的测试结果
* cnn.py: CNN模型的python代码
* data.py: 处理数据的python代码
* knn.cpp: KNN模型代码
* knn.csv: KNN模型得出的测试结果
* navie.py: 全连接模型的python代码
* parse.py: 将数据处理成图片
* sample_submission.csv: 样例提交代码
* test.csv: 测试数据
* train.csv: 训练数据

## CNN

```sh
> python cnn.py
```

## KNN

```sh
> g++ knn.cpp -o knn.exe --std=c++11 -O2
> ./knn.exe
```
