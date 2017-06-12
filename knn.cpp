#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

const int WIDTH = 28, HEIGHT = 28, K = 3;
struct IMG
{
    int data[WIDTH*HEIGHT];
};

char buf[1024*100];
vector<IMG> train_imgs;
vector<int> train_labels;
vector<IMG> predict_imgs;

int readInt(FILE *fd)
{
    int rst=0;
    char c;
    do {
        c = fgetc(fd);
        if (c == EOF) return -1;
    } while(c < '0' || '9' < c);
    do {rst = rst*10 + c - '0'; c = fgetc(fd);} while('0' <= c && c <= '9');
    return rst;
}

int knn(const IMG &input, int size = -1)
{
    if(size < 0) size = train_imgs.size();
    size = min(size, (int)train_imgs.size());

    int indexs[K];
    int min_dis[K];
    for(int i = 0; i < K; i ++)
    {
        indexs[i] = -1;
        min_dis[i] = 0x7fffffff/2;
    }

    for(int i = 0; i < size; i ++)
    {
        int tmp_dis = 0;
        for(int j = 0; j < WIDTH*HEIGHT; j ++)
            tmp_dis += (input.data[j]-train_imgs[i].data[j])*(input.data[j]-train_imgs[i].data[j]);
        int g;
        for(g = 0; g < K && min_dis[g] <= tmp_dis; g ++);
        for(int j = K-1; j > g; j --) {
            indexs[j] = indexs[j-1];
            min_dis[j] = min_dis[j-1];
        }
        if (g < K)
        {
            min_dis[g] = tmp_dis;
            indexs[g] = i;
        }
    }

    int best_count = 0, best_value, count = 0, value = -1;
    int labels[K];
    for(int i = 0; i < K; i ++)
        labels[i] = train_labels[indexs[i]];
    sort(labels, labels+K);
    for(int i=0;i<K;i++)
    {
        if(labels[i] == value)
        {
            count ++;
        } else {
            count = 1;
            value = labels[i];
        }
        if (best_count < count)
        {
            best_count = count;
            best_value = value;
        }
    }
    return best_value;
}

int main()
{
    {
        FILE *fd = fopen("train.csv", "r");
        fgets(buf, sizeof(buf), fd);
        int label = 0;
        IMG img;
        while((label = readInt(fd)) >= 0)
        {
            for(int i = 0; i < WIDTH*HEIGHT; i ++)
                img.data[i] = readInt(fd);
            train_labels.push_back(label);
            train_imgs.push_back(img);
        }

        fclose(fd);

        cout << "train size : " << train_imgs.size() << endl;
    }

    {
        FILE *fd = fopen("test.csv", "r");
        fgets(buf, sizeof(buf), fd);
        IMG img;
        while((img.data[0] = readInt(fd)) >= 0)
        {
            for(int i = 1; i < WIDTH*HEIGHT; i ++)
                img.data[i] = readInt(fd);
            predict_imgs.push_back(img);
        }

        fclose(fd);

        cout << "predict size : " << predict_imgs.size() << endl;
    }

    int TRAIN_SIZE = train_imgs.size()*0.99;
    int A = 0, B = 0;
    for(int i = TRAIN_SIZE; i < (int)train_imgs.size(); i ++)
    {
        cout << "training : " << i << endl;
        B ++;
        if (knn(train_imgs[i], TRAIN_SIZE) == train_labels[i]) A ++;
    }
    cout << A << " / " << B << " " << double(A)/B << endl;

    {
        FILE *fd = fopen("knn.csv", "w");
        fprintf(fd, "ImageId,Label\n");
        for(int i = 0; i < (int)predict_imgs.size(); i ++)
        {
            cout << "predicting : " << i << endl;
            fprintf(fd, "%d,%d\n", i+1, knn(predict_imgs[i]));
        }
        fclose(fd);
    }

    return 0;
}