#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <opencv2/core/types_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "inference.h"
#include "QImage"
#include "QPainter"
#include "QFileDialog"
#include "QElapsedTimer"
#include "QJsonObject"
#include "QJsonDocument"
using namespace cv;
using namespace std;
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}
Inference Infer;
QJsonObject Dataobj;

QJsonObject JsonLoad(QString jsonpath)
{
    QFile jtxtfile(jsonpath);
    jtxtfile.open(QFile::ReadOnly);
    QByteArray barr = jtxtfile.readAll();
    QJsonObject Jobj=QJsonDocument::fromJson(barr).object();
    return Jobj;
}

void MainWindow::on_pushButton_clicked()
{
    Infer.LoadModel("/home/gns/Documents/SamDoc/Final/saved_model");
    Dataobj = JsonLoad("Data.json");
}


QPainter *qPainter;
void MainWindow::on_pushButton_2_clicked()
{
    try
    {
        QString path =
                QFileDialog::getOpenFileName(this, "Open a file", "directoryToOpen",
                                             "Images (*.png *.xpm *.jpg)");
        QElapsedTimer qt_cyc;
        qt_cyc.start();
        Results RawRes=Infer.Predict(path.toStdString());
        QImage PreImage(path);
        vector<Rect> Boxes{};
        vector<float> Scores{};
        vector<int> ResBoxes{};
        for(size_t i=0;i<300;++i)
        {
            Rect Box(RawRes.boxes[1 + i*4]*PreImage.width(),RawRes.boxes[0 + i*4]*PreImage.height(),(RawRes.boxes[3+i*4]-RawRes.boxes[1+i*4])*PreImage.width(),(RawRes.boxes[2+i*4]-RawRes.boxes[0+i*4])*PreImage.height());
            Boxes.push_back(Box);
            Scores.push_back(RawRes.scores[i]);

        }

        dnn::NMSBoxes(Boxes,Scores,0.5,0.5,ResBoxes);
        cout<<"Result Box count "<<ResBoxes.size()<<endl;
        int AppleCnt=0;
        int BananaCnt=0;
        int OrangeCnt=0;
        for(auto res:ResBoxes)
        {
            QString lab= QString::number(RawRes.label_ids[res]);
            QString FruitePredected=Dataobj[lab].toString();
            if(lab=="1")
            {
                AppleCnt++;
            }
            else if(lab=="2")
            {
                BananaCnt++;
            }
            else if(lab=="3")
            {
                OrangeCnt++;
            }
            qPainter = new QPainter(&PreImage);
            qPainter->setPen(QPen(Qt::red,3,Qt::SolidLine));
            qPainter->drawRect(RawRes.boxes[1+res*4]*PreImage.width(),RawRes.boxes[0+res*4]*PreImage.height(),(RawRes.boxes[3+res*4]-RawRes.boxes[1+res*4])*PreImage.width(),(RawRes.boxes[2+res*4]-RawRes.boxes[0+res*4])*PreImage.height());
            qPainter->setFont(QFont("Courier",5));
            qPainter->drawText(RawRes.boxes[1 + res*4]*PreImage.width(),RawRes.boxes[0 + res*4]*PreImage.height()-20,FruitePredected);
            qPainter->end();
            delete qPainter;
        }
        ui->label->setScaledContents(true);
        ui->label->setSizePolicy(QSizePolicy::Ignored,QSizePolicy::Ignored);
        ui->label->setPixmap(QPixmap::fromImage(PreImage));
        ui->label_Aple->setNum(AppleCnt);
        ui->label_Ban->setNum(BananaCnt);
        ui->label_org->setNum(OrangeCnt);
        int tm_cyl=qt_cyc.elapsed();
        QVariant qv_cyl(tm_cyl);
        ui->label_2->setText(qv_cyl.toString()+" ms");
    }
    catch (...)
    {
        cout<<"Error in Post Processing"<<endl;
    }
}
