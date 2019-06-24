# -*- coding: utf-8 -*-
from PyQt4.QtCore import *
from skimage import io,color
import numpy as np
from PIL import Image
from PyQt4.QtGui import *
from PyQt4.QtCore import pyqtSlot,SIGNAL,SLOT
from PyQt4 import QtCore, QtGui
from teztasarim import Ui_Dialog
from scipy import ndimage
from skimage import filters
from skimage.filters import threshold_otsu
from skimage import img_as_uint
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import xlwt
from sklearn import preprocessing

class MainWindow(QtGui.QMainWindow, Ui_Dialog):  
    global Logistic_Regression,Naive_Bayes,SVM,KNN,Random_Forest,ayarla
    
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setupUi(self)
        
        self.connect(self.t2_btn_test_yuzde_ayarla,QtCore.SIGNAL('clicked()'),self.test_yuzde_ayarla)
        self.connect(self.t1_btn_veriyukle,QtCore.SIGNAL('clicked()'),self.datayukle)
        #self.connect(self.tableWidget, QtCore.SIGNAL('itemChanged(QTableWidgetItem*)'),self.datayukle)
        self.connect(self.t2_btn_egit,QtCore.SIGNAL('clicked()'),self.egitim)
        self.connect(self.t1_btn_data_listele,QtCore.SIGNAL('clicked()'),self.data_listele)        
        self.connect(self.t1_btn_excel,QtCore.SIGNAL('clicked()'),self.excele_donustur)
        self.connect(self.t2_btn_veriseti_listele,QtCore.SIGNAL('clicked()'),self.veri_goster)        
        self.connect(self.t3_btn_normalize_goster,QtCore.SIGNAL('clicked()'),self.normalize_goster)
        self.connect(self.t3_btn_egit,QtCore.SIGNAL('clicked()'),self.egitim_normalize)
        self.connect(self.t3_btn_normalize_et,QtCore.SIGNAL('clicked()'),self.normalize_et)
        self.connect(self.t4_btn_karsilastir,QtCore.SIGNAL('clicked()'),self.karsilastir)
        
#        data=pandas.read_excel('./breast-cancer-wisconsin2.xls')
#        data=np.array(data)
#        column=data.shape[1]
#        satir=data.shape[0]
#        self.tableWidget_2.setColumnCount(column-1)
#        self.tableWidget_2.setRowCount(satir+1) ##set number of rows
#        self.tableWidget_2.setHorizontalHeaderLabels(['Kitle Kalinligi', 'Uniform Hucre Boyutu', 'Uniform Hucre Sekli', 'Marjinal Adhezyon', 'Tek Epitel Hucre Boyutu','Sitoplazma Icerme Durumu','Yumusak Kromatin','Normal Çekirdekçik','Mitoz','Sinif: 2->Iyi Huylu'"\n"' 4->Kotu Huylu'])
#        for rowNumber,row in enumerate(data):
#            for i in range(0,len(row)-1):
#                self.tableWidget_2.setItem(rowNumber, i, QtGui.QTableWidgetItem(str(row[i+1])))
    
    def karsilastir(self):
        print "a"
    def normalize_et(self):
        book = xlwt.Workbook(encoding="utf-8")
        sheet1 = book.add_sheet("Sheet 1")
        park=pandas.read_excel('./breast-cancer-wisconsin2.xls')
        df=pandas.DataFrame(park)
        min_scaler=preprocessing.MinMaxScaler()
        np_scaled=min_scaler.fit_transform(df)
        df_normalized=pandas.DataFrame(np_scaled)
        #print df_normalized[1][0]#sütun yazdırıyor.df_normalized[9][8]=9.sütunun 8.satırını yazdırır.
        sheet1.write(0,0,'Kitle Kalinligi')
        sheet1.write(0,1,'Uniform Hucre Boyutu')
        sheet1.write(0,2,'Uniform Hucre Sekli')
        sheet1.write(0,3,'Marjinal Adhezyon')
        sheet1.write(0,4,'Tek Epitel Hucre Boyutu')
        sheet1.write(0,5,'Sitoplazma Icerme Durumu')
        sheet1.write(0,6,'Yumusak Kromatin')
        sheet1.write(0,7,'Normal Çekirdekçik')
        sheet1.write(0,8,'Mitoz')
        sheet1.write(0,9,'Sinif')

        for sutun in range(0,10):
            ilk=0
            for satir in range(1,684):    
                sheet1.write(satir,sutun,df_normalized[sutun+1][ilk])
                ilk+=1
                if ilk==683:
                    break       
        book.save("./Dataset_normalize.xls")
        self.label_7.setText("Veri seti normalize edildi.")
    def normalize_goster(self):
        data=pandas.read_excel('./Dataset_normalize.xls')
        print (data.shape)
        data=np.array(data)
        column=data.shape[1]
        satir=data.shape[0]
        print column,satir
        
        self.t3_tableWidget.setColumnCount(column)
        self.t3_tableWidget.setRowCount(satir) ##set number of rows

        self.t3_tableWidget.setHorizontalHeaderLabels(['Kitle Kalinligi', 'Uniform Hucre Boyutu', 'Uniform Hucre Sekli', 'Marjinal Adhezyon', 'Tek Epitel Hucre Boyutu','Sitoplazma Icerme Durumu','Yumusak Kromatin','Normal Çekirdekçik','Mitoz','Sinif: 0->Iyi Huylu'"\n"' 1->Kotu Huylu'])

        for rowNumber,row in enumerate(data):
            for i in range(0,len(row)):
                
                self.t3_tableWidget.setItem(rowNumber, i, QtGui.QTableWidgetItem(str(row[i])))
    
        
    def egitim(self):
        dataset='./breast-cancer-wisconsin2.xls'
        islemCode=self.cb_algoritmalar.currentIndex()
        if islemCode==0:
            Logistic_Regression(self,dataset)
            
        elif islemCode==1:
            Naive_Bayes(self,dataset)
        elif islemCode==2:
            SVM(self,dataset)
        elif islemCode==3:
            KNN(self,dataset)
        else:
            Random_Forest(self,dataset)
       
    def egitim_normalize(self):
        veriseti='./Dataset_normalize.xls'
        islemCode=self.cb2_algoritmalar.currentIndex()
        if islemCode==0:
            Logistic_Regression(self,veriseti)
            
        elif islemCode==1:
            Naive_Bayes(self,veriseti)
        elif islemCode==2:
            SVM(self,veriseti)
        elif islemCode==3:
            KNN(self,veriseti)
        else:
            Random_Forest(self,veriseti)   
    def veri_goster(self):
        data=pandas.read_excel('./breast-cancer-wisconsin2.xls')
        print (data.shape)
        data=np.array(data)
        column=data.shape[1]
        satir=data.shape[0]
        print column,satir
        
        self.tableWidget_2.setColumnCount(column)
        self.tableWidget_2.setRowCount(satir) ##set number of rows

        self.tableWidget_2.setHorizontalHeaderLabels(['ID','Kitle Kalinligi', 'Uniform Hucre Boyutu', 'Uniform Hucre Sekli', 'Marjinal Adhezyon', 'Tek Epitel Hucre Boyutu','Sitoplazma Icerme Durumu','Yumusak Kromatin','Normal Çekirdekçik','Mitoz','Sinif: 2->Iyi Huylu'"\n"' 4->Kotu Huylu'])

        for rowNumber,row in enumerate(data):
            for i in range(0,len(row)):
                self.tableWidget_2.setItem(rowNumber, i, QtGui.QTableWidgetItem(str(row[i])))
    def datayukle(self):
        data=pandas.read_excel('./breast-cancer-wisconsin2.xls')
        print (data.shape)
        data=np.array(data)
        column=data.shape[1]
        satir=data.shape[0]
        print column,satir
        
        self.t1_tableWidget.setColumnCount(column)
        self.t1_tableWidget.setRowCount(satir+1) ##set number of rows

        self.t1_tableWidget.setHorizontalHeaderLabels(['ID','Kitle Kalinligi', 'Uniform Hucre Boyutu', 'Uniform Hucre Sekli', 'Marjinal Adhezyon', 'Tek Epitel Hucre Boyutu','Sitoplazma Icerme Durumu','Yumusak Kromatin','Normal Çekirdekçik','Mitoz','Sinif: 2->Iyi Huylu'"\n"' 4->Kotu Huylu'])

        for rowNumber,row in enumerate(data):
            for i in range(0,len(row)):
                self.t1_tableWidget.setItem(rowNumber, i, QtGui.QTableWidgetItem(str(row[i])))
                
   
    def data_listele(self):
        f = open('./breast-cancer-wisconsin (1).data')
        for i,line in enumerate(f.readlines()):
            self.t1_listWidget.addItem(line)
        with open('./breast-cancer-wisconsin (1).data') as f:
            print ("Toplam:",sum(1 for _ in f))

        f.close()
         
    def excele_donustur(self):
        book = xlwt.Workbook(encoding="utf-8")
        sheet1 = book.add_sheet("Sheet 1")

        f = open('./breast-cancer-wisconsin (1).data')
        row2=1
        sheet1.write(0,0,'ID')
        sheet1.write(0,1,'Kitle Kalinligi')
        sheet1.write(0,2,'Uniform Hucre Boyutu')
        sheet1.write(0,3,'Uniform Hucre Sekli')
        sheet1.write(0,4,'Marjinal Adhezyon')
        sheet1.write(0,5,'Tek Epitel Hucre Boyutu')
        sheet1.write(0,6,'Sitoplazma Icerme Durumu')
        sheet1.write(0,7,'Yumusak Kromatin')
        sheet1.write(0,8,'Normal Çekirdekçik')
        sheet1.write(0,9,'Mitoz')
        sheet1.write(0,10,'Sinif')
        for i,line in enumerate(f.readlines()):
            kayitYap=True
            currentline = line.split(",")
            print currentline
    
            for m in range(0,len(currentline)):
                if currentline[m]=='?' or currentline[m]==None or currentline[m]==' ' or currentline[m]=="" or currentline[m]==" ":
                    kayitYap=False #.data uzantılı indirlen veri setinde herhangi bir yanlışlık olursa o satırı excel dosyasına kaydetmiyor.
  
            if kayitYap==True:
                for column_no in range(0,len(currentline)):
                        sheet1.write(row2,column_no, currentline[column_no])     
                row2+=1
        with open('./breast-cancer-wisconsin (1).data') as f:
            print ("Toplam:",sum(1 for _ in f),row2)
        f.close()
        book.save("./breast-cancer-wisconsin2.xls")
    
    def ayarla(self):
        self.t4_tableWidget.setColumnCount(2)
        self.t4_tableWidget.setRowCount(10)
        
        self.t4_tableWidget.setHorizontalHeaderLabels(['Algoritma','Basarisi'])
    def test_yuzde_ayarla(self):
        test_degeri=float(self.t2_line_Test.text())
        egitim_degeri=str(1.00-test_degeri)
        self.t2_line_egitim.setText(egitim_degeri)
        return test_degeri
    def Random_Forest(self,okunan_veri):
        
        k=5

        data=pandas.read_excel(okunan_veri)
        data=np.array(data)
        X=data[:,1:data.shape[1]-1]
        y=data[:,data.shape[1]-1]
        
        print (X.shape,len(y))


      
      
        rs = ShuffleSplit(n_splits=k, test_size=0.20, random_state=0)
        rs.get_n_splits(X)
        
        clf_RF = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
        
        genel_acc_RF=0
        sayac=0
        for train_index, test_index in rs.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train=[]
            X_test=[]
            y_train=[]
            y_test=[]
   
            for idx_no in train_index:
                X_train.append(X[idx_no])
                y_train.append(y[idx_no])

            for idx_no in test_index:
                X_test.append(X[idx_no])
                y_test.append(y[idx_no])
      
            X_train=np.array(X_train)
            X_test=np.array(X_test)
            #print ("X train:",X_train.shape," X test:",X_test.shape)
       
      
            clf_RF.fit(X_train,y_train)
            results=clf_RF.predict(X_test)
            cm=confusion_matrix(y_test,results)
            print ("Conf matrix:",cm[0][1])
            acc=(float(cm[0][0])+cm[1][1])/(cm[0][1]+cm[1][0]+cm[0][0]+cm[1][1])
            acc2=accuracy_score(y_test,results)
            genel_acc_RF+=acc
            sayac+=1
            print ("k degeri:",sayac," Acc value:",acc*100," Acc 2:",acc2*100)       
        #print ("RF Ortalama Acc:",round((float(genel_acc_RF)/k)*100,3))
        ort=round((float(genel_acc_RF)/k)*100,3)
        ort=str(ort)
        if okunan_veri=='./breast-cancer-wisconsin2.xls':
                       
            self.t2_listWidget.addItem("Random Forest ile K-Fold Ortalama: "+ort)
            ayarla(self)
            self.t4_tableWidget.setItem(2, 0, QtGui.QTableWidgetItem(str("Random Forest Ortalama")))
            self.t4_tableWidget.setItem(2, 1, QtGui.QTableWidgetItem(ort))
            
            basari=self.t4_tableWidget.item(2,1).text()
            
            
            
        else:
            self.t3_listWidget.addItem("Normalize Random Forest ile K-Fold Ortalama: "+ort)
            self.t4_tableWidget.setItem(3, 0, QtGui.QTableWidgetItem(str("Normalize Random Forest Ortalama")))
            self.t4_tableWidget.setItem(3, 1, QtGui.QTableWidgetItem(ort))
            basari=self.t4_tableWidget.item(3,1).text()
            
            
    def KNN(self,okunan_veri):
        k=5

        data=pandas.read_excel(okunan_veri)
        data=np.array(data)
        X=data[:,1:data.shape[1]-1]
        y=data[:,data.shape[1]-1]
        print data[0:1]
        #print (X.shape,len(y))



        rs = ShuffleSplit(n_splits=k, test_size=.20, random_state=0)
        rs.get_n_splits(X)
        
        clf_KNN = KNeighborsClassifier(n_neighbors=3)
        
        genel_acc_KNN=0
        sayac=0
        for train_index, test_index in rs.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train=[]
            X_test=[]
            y_train=[]
            y_test=[]
   
            for idx_no in train_index:
                X_train.append(X[idx_no])
                y_train.append(y[idx_no])

            for idx_no in test_index:
                X_test.append(X[idx_no])
                y_test.append(y[idx_no])
      
            X_train=np.array(X_train)
            X_test=np.array(X_test)
            #print ("X train:",X_train.shape," X test:",X_test.shape)
       
      
            clf_KNN.fit(X_train,y_train)
            results=clf_KNN.predict(X_test)
            cm=confusion_matrix(y_test,results)
            #print ("Conf matrix:",cm[0][1])
            acc=(float(cm[0][0])+cm[1][1])/(cm[0][1]+cm[1][0]+cm[0][0]+cm[1][1])
            acc2=accuracy_score(y_test,results)
            genel_acc_KNN+=acc
            sayac+=1
#            print ("k degeri:",sayac," Acc value:",acc*100," Acc 2:",acc2*100)
#        print ("KNN Ortalama Acc:",round((float(genel_acc_KNN)/k)*100,3))
        ort=round((float(genel_acc_KNN)/k)*100,3)
        ort=str(ort)
        if okunan_veri=='./breast-cancer-wisconsin2.xls':
            self.t2_listWidget.addItem("KNN ile K-Fold Ortalama: "+ort)
            ayarla(self)
            self.t4_tableWidget.setItem(0, 0, QtGui.QTableWidgetItem(str("KNN Ortalama")))
            self.t4_tableWidget.setItem(0, 1, QtGui.QTableWidgetItem(ort))
        else:
            self.t3_listWidget.addItem("Normalize KNN ile K-Fold Ortalama: "+ort)
            self.t4_tableWidget.setItem(1, 0, QtGui.QTableWidgetItem(str("Normalize KNN Ortalama")))
            self.t4_tableWidget.setItem(1, 1, QtGui.QTableWidgetItem(ort))
            
    def SVM(self,okunan_veri):
        k=5

        data=pandas.read_excel(okunan_veri)
        data=np.array(data)
        X=data[:,1:data.shape[1]-1]
        y=data[:,data.shape[1]-1]

        print (X.shape,len(y))



        rs = ShuffleSplit(n_splits=k, test_size=.20, random_state=0)
        rs.get_n_splits(X)
        
        clf_SVM = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                          decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                          max_iter=-1, probability=False, random_state=None, shrinking=True,
                          tol=0.001, verbose=False)
        
        genel_acc_SVM=0
        sayac=0
        for train_index, test_index in rs.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train=[]
            X_test=[]
            y_train=[]
            y_test=[]
   
            for idx_no in train_index:
                X_train.append(X[idx_no])
                y_train.append(y[idx_no])

            for idx_no in test_index:
                X_test.append(X[idx_no])
                y_test.append(y[idx_no])
      
            X_train=np.array(X_train)
            X_test=np.array(X_test)
            print ("X train:",X_train.shape," X test:",X_test.shape)
       
      
            clf_SVM.fit(X_train,y_train)
            results=clf_SVM.predict(X_test)
            cm=confusion_matrix(y_test,results)
            print ("Conf matrix:",cm[0][1])
            acc=(float(cm[0][0])+cm[1][1])/(cm[0][1]+cm[1][0]+cm[0][0]+cm[1][1])
            acc2=accuracy_score(y_test,results)
            genel_acc_SVM+=acc
            sayac+=1
            print ("k degeri:",sayac," Acc value:",acc*100," Acc 2:",acc2*100)
        #print ("SVM Ortalama Acc:",round((float(genel_acc_SVM)/k)*100,3))
        ort=round((float(genel_acc_SVM)/k)*100,3)
        ort=str(ort)
        if okunan_veri=='./breast-cancer-wisconsin2.xls':
            self.t2_listWidget.addItem("SVM ile K-Fold Ortalama: "+ort)
            ayarla(self)
            self.t4_tableWidget.setItem(4, 0, QtGui.QTableWidgetItem(str("SVM Ortalama")))
            self.t4_tableWidget.setItem(4, 1, QtGui.QTableWidgetItem(ort))
        else:
            self.t3_listWidget.addItem("Normalize SVM ile K-Fold Ortalama: "+ort)
            self.t4_tableWidget.setItem(5, 0, QtGui.QTableWidgetItem(str("Normolize SVM Ortalama")))
            self.t4_tableWidget.setItem(5, 1, QtGui.QTableWidgetItem(ort))
        
        
    def Logistic_Regression(self,okunan_veri):
        k=5

        data=pandas.read_excel(okunan_veri)
        data=np.array(data)
        X=data[:,1:data.shape[1]-1]
        y=data[:,data.shape[1]-1]

        print (X.shape,len(y))



        rs = ShuffleSplit(n_splits=k, test_size=.20, random_state=0)
        rs.get_n_splits(X)
        
        clf_lr = LogisticRegression()

        genel_acc_lr=0
        sayac=0
        for train_index, test_index in rs.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train=[]
            X_test=[]
            y_train=[]
            y_test=[]
   
            for idx_no in train_index:
                X_train.append(X[idx_no])
                y_train.append(y[idx_no])

            for idx_no in test_index:
                X_test.append(X[idx_no])
                y_test.append(y[idx_no])
      
            X_train=np.array(X_train)
            X_test=np.array(X_test)
            print ("X train:",X_train.shape," X test:",X_test.shape)
       
      
            clf_lr.fit(X_train,y_train)
            results=clf_lr.predict(X_test)
            cm=confusion_matrix(y_test,results)
            print ("Conf matrix:",cm[0][1])
            acc=(float(cm[0][0])+cm[1][1])/(cm[0][1]+cm[1][0]+cm[0][0]+cm[1][1])
            acc2=accuracy_score(y_test,results)
            genel_acc_lr+=acc
            sayac+=1
            print ("k degeri:",sayac," Acc value:",acc*100," Acc 2:",acc2*100)
        #print ("LR Ortalama Acc:",round((float(genel_acc_lr)/k)*100,3))
        ort=round((float(genel_acc_lr)/k)*100,3)
        ort=str(ort)
        if okunan_veri=='./breast-cancer-wisconsin2.xls':
            self.t2_listWidget.addItem("Logistic Regresyon ile K-Fold Ortalama: "+ort)
            ayarla(self)
            self.t4_tableWidget.setItem(6, 0, QtGui.QTableWidgetItem(str("Logistic Regresyon Ortalama")))
            self.t4_tableWidget.setItem(6, 1, QtGui.QTableWidgetItem(ort))
        else:
            self.t3_listWidget.addItem("Normalize Logistic Regresyon ile K-Fold Ortalama: "+ort)
            self.t4_tableWidget.setItem(7, 0, QtGui.QTableWidgetItem(str("Normalize Logistic Regresyon Ortalama")))
            self.t4_tableWidget.setItem(7, 1, QtGui.QTableWidgetItem(ort))
       
        
        
    
        
    def Naive_Bayes(self,okunan_veri):
        k=5

        data=pandas.read_excel(okunan_veri)
        data=np.array(data)
        X=data[:,1:data.shape[1]-1]
        y=data[:,data.shape[1]-1]

        print (X.shape,len(y))



        rs = ShuffleSplit(n_splits=k, test_size=.20, random_state=0)
        rs.get_n_splits(X)
        
    
        clf_NB=GaussianNB()
        genel_acc_NB=0
        sayac=0
        for train_index, test_index in rs.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train=[]
            X_test=[]
            y_train=[]
            y_test=[]
   
            for idx_no in train_index:
                X_train.append(X[idx_no])
                y_train.append(y[idx_no])

            for idx_no in test_index:
                X_test.append(X[idx_no])
                y_test.append(y[idx_no])
      
            X_train=np.array(X_train)
            X_test=np.array(X_test)
            print ("X train:",X_train.shape," X test:",X_test.shape)
       
      
            clf_NB.fit(X_train,y_train)
            results=clf_NB.predict(X_test)
            cm=confusion_matrix(y_test,results)
            print ("Conf matrix:",cm[0][1])
            acc=(float(cm[0][0])+cm[1][1])/(cm[0][1]+cm[1][0]+cm[0][0]+cm[1][1])
            acc2=accuracy_score(y_test,results)
            genel_acc_NB+=acc
            sayac+=1
            print ("k degeri:",sayac," Acc value:",acc*100," Acc 2:",acc2*100)
        #print ("Naive Bayes Ortalama Acc:",round((float(genel_acc_NB)/k)*100,3))
        ort=round((float(genel_acc_NB)/k)*100,3)
        ort=str(ort)
        if okunan_veri=='./breast-cancer-wisconsin2.xls':
            self.t2_listWidget.addItem("Naive Bayes ile K-Fold Ortalama: "+ort)
            ayarla(self)
            self.t4_tableWidget.setItem(8, 0, QtGui.QTableWidgetItem(str("Naive Bayes Ortalama")))
            self.t4_tableWidget.setItem(8, 1, QtGui.QTableWidgetItem(ort))
        else:
            self.t3_listWidget.addItem("Normalize Naive Bayes ile K-Fold Ortalama: "+ort)
            ayarla(self)
            self.t4_tableWidget.setItem(9, 0, QtGui.QTableWidgetItem(str("Normalize Naive Bayes Ortalama")))
            self.t4_tableWidget.setItem(9, 1, QtGui.QTableWidgetItem(ort))
        

            
            
            
            