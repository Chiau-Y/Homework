from PyQt5 import QtWidgets, uic,QtCore,QtGui
from PyQt5.QtWidgets import QTableWidget,QTableWidgetItem
import pymysql
import os
import sys

# DBMS setting (host, user, password, db)
db = pymysql.connect('localhost','root','root',"property_schema")
cursor = db.cursor()

# GUI setting
path = os.getcwd()
qtCreatorFile = path + os.sep + "Hw.ui"
Ui_Hw, QtBaseClass = uic.loadUiType(qtCreatorFile)
_translate = QtCore.QCoreApplication.translate

# ------------------------------------Main program-----------------------------------
class MainUi(QtWidgets.QTabWidget, Ui_Hw):
    def __init__(self):
        QtWidgets.QTabWidget.__init__(self)
        Ui_Hw.__init__(self)
        self.setupUi(self)
        
        # Button for trigger
        self.P1_check.clicked.connect(self.Data_Show) 
        self.P3_insert.clicked.connect(self.Insert) 
        self.P3_delete.clicked.connect(self.Delete) 
        self.P3_salary.clicked.connect(self.Increasing_salary) 
        self.P2_check.clicked.connect(self.Data_people_lab) 
        self.P2_project.clicked.connect(self.Project_lab)
        self.P2_PhD.clicked.connect(self.PhD_lab)
        self.P2_student.clicked.connect(self.Project_student)
        self.P2_check_2.clicked.connect(self.Project_funding)        
        self.P2_maxmin.clicked.connect(self.Project_maxmin)
        self.P2_avg.clicked.connect(self.Project_avg)
        self.P4_check.clicked.connect(self.Data_coding)
        
# ------------------------------------ Subroutine ------------------------------------    
    def Data_Show(self):   
    # Import the data in DB
        x = str(self.P1_combobox.currentText())
        self.tableWidget.clear()
        # Call table
        if x == "LAB":
            comman_data =  """SELECT* FROM LAB"""
            cursor.execute(comman_data)
            data = list(cursor.fetchall())
            self.tableWidget.setRowCount(len(data))
            self.tableWidget.setColumnCount(len(data[0]))
            self.tableWidget.setHorizontalHeaderLabels(["LAB_No.","Name","Property"])
        elif x == "Teacher":    
            comman_data =  """SELECT* FROM Teacher"""
            cursor.execute(comman_data)
            data = list(cursor.fetchall())            
            self.tableWidget.setRowCount(len(data))
            self.tableWidget.setColumnCount(len(data[0]))
            self.tableWidget.setHorizontalHeaderLabels(["ID","Name","Start_Date","LAB_No."])
        elif x == "Student":
            comman_data =  """SELECT* FROM Student"""
            cursor.execute(comman_data)
            data = list(cursor.fetchall())            
            self.tableWidget.setRowCount(len(data))
            self.tableWidget.setColumnCount(len(data[0]))
            self.tableWidget.setHorizontalHeaderLabels(["ID","Name","Degree","Salary","LAB_No."])
        elif x == "Equipment":
            comman_data =  """SELECT* FROM Equipment"""
            cursor.execute(comman_data)
            data = list(cursor.fetchall())            
            self.tableWidget.setRowCount(len(data))
            self.tableWidget.setColumnCount(len(data[0]))
            self.tableWidget.setHorizontalHeaderLabels(["No.","Cost","Buy_Date","LAB_No."])
        elif x == "MOST_Project":
            comman_data =  """SELECT* FROM MOST_Project"""
            cursor.execute(comman_data)
            data = list(cursor.fetchall())            
            self.tableWidget.setRowCount(len(data))
            self.tableWidget.setColumnCount(len(data[0]))
            self.tableWidget.setHorizontalHeaderLabels(["No.","Start_Date","End_Date","Funding"])
        elif x == "Execute":
            comman_data =  """SELECT* FROM Execute"""
            cursor.execute(comman_data)
            data = list(cursor.fetchall())            
            self.tableWidget.setRowCount(len(data))
            self.tableWidget.setColumnCount(len(data[0]))
            self.tableWidget.setHorizontalHeaderLabels(["ID_Teacher","ID_Student","Project_No."])
        elif x == "Accounting_of":
            comman_data =  """SELECT* FROM Accounting_of"""
            cursor.execute(comman_data)
            data = list(cursor.fetchall())            
            self.tableWidget.setRowCount(len(data))
            self.tableWidget.setColumnCount(len(data[0]))
            self.tableWidget.setHorizontalHeaderLabels(["ID","LAB_No.","Start_Date"])            
        font = self.tableWidget.horizontalHeader().font()
        font.setBold(True)
        self.tableWidget.horizontalHeader().setFont(font)
        # Import data from the table to GUI
        for i in range (len(data)):
            for j in range (len(data[0])):
                self.tableWidget.setItem(i,j,QTableWidgetItem(str(data[i][j])))    
    def Insert(self): 
    # Insert the data of the equipment //INSERT
        e_no = str(self.P3_No.text())
        e_cost = str(self.P3_cost.text())
        e_date = str(self.P3_date.text())
        e_lab = str(self.P3_lab.text())  
        command_Add_Equip = 'INSERT INTO Equipment(E_No, Cost, Buy_Date, LAB_No) VALUES (\'' + str(e_no) + '\', \'' + str(e_cost) + '\', \'' + str(e_date) + '\',\'' + str(e_lab) + '\')'
        cursor.execute(command_Add_Equip)
        db.commit()  
        self.P3_No.clear()
        self.P3_cost.clear() 
        self.P3_date.clear()
        self.P3_lab.clear()
    def Delete(self):
    # Delete the data of the equipment //DELETE
        e_no_2 = str(self.P3_No_2.text())
        command_Delete_Equip = 'DELETE FROM Equipment WHERE E_No = \'' + str(e_no_2) + '\''
        cursor.execute(command_Delete_Equip) 
        db.commit()    
        self.P3_No_2.clear()    
    def Increasing_salary(self):
    # Increasing someone's salary who works for acconting or MOST_project // UPDATE, IN  
        salary = str(self.P3_salaryadd.text()) 
        if int(self.P3_checkBox.checkState()) == 2 :
            # For both
            if int(self.P3_checkBox_2.checkState()) == 2:                               
                command_Increase_Salary = 'UPDATE Student SET Salary = Salary + ' + \
                    str(salary) +' WHERE ID_S IN (SELECT ID_Student FROM Execute, Accounting_of WHERE ID_Student = ID_Accounting)'    
            else:
                # Only for accounting
                command_Increase_Salary = 'UPDATE Student SET Salary = Salary + ' + \
                    str(salary) +' WHERE ID_S IN (SELECT ID_S FROM Accounting_of WHERE ID_S = ID_Accounting)'  
        else:
            # Only for project
            if int(self.P3_checkBox_2.checkState()) == 2:
                command_Increase_Salary = 'UPDATE Student SET Salary = Salary + ' + \
                    str(salary) +' WHERE ID_S IN (SELECT ID_S FROM Execute WHERE ID_Student = ID_S)'                   
            else:
                pass                    
        cursor.execute(command_Increase_Salary)
        db.commit() 
        self.P3_salaryadd.clear()
    def Data_people_lab(self):
    # The number of the people in the LAB and the total of the salary // COUNT, SUM
        command_count_Student_Salary = """SELECT L_No, COUNT(*), SUM(Salary) 
                                        FROM Student , LAB 
                                        WHERE LAB_No = L_No GROUP BY LAB_No"""   
        cursor.execute(command_count_Student_Salary)
        data = list(cursor.fetchall())  
        self.P2_tableWidget.setRowCount(len(data))
        self.P2_tableWidget.setColumnCount(len(data[0]))
        self.P2_tableWidget.setHorizontalHeaderLabels(["LAB_No.","Number of people","Sum of Salary"])        
        font = self.P2_tableWidget.horizontalHeader().font()
        font.setBold(True)
        self.P2_tableWidget.horizontalHeader().setFont(font)
        # Import data to GUI
        for i in range (len(data)):
            for j in range (len(data[0])):
                self.P2_tableWidget.setItem(i,j,QTableWidgetItem(str(data[i][j])))                                             
    def Project_lab(self):
    # Which Lab has MOST_project // EXISTS
        self.tableWidget.clear()
        command_Funding_Lab = """SELECT L_No
                                FROM LAB, Teacher
                                WHERE EXISTS (SELECT * 
                                              FROM Execute
                                              WHERE ID_Teacher = ID_T AND LAB_No = L_No)"""  
        cursor.execute(command_Funding_Lab)
        data = list(cursor.fetchall())  
        self.P2_tableWidget_2.setRowCount(len(data))
        self.P2_tableWidget_2.setColumnCount(len(data[0]))
        self.P2_tableWidget_2.setHorizontalHeaderLabels(["LAB_No."]) 
        font = self.P2_tableWidget_2.horizontalHeader().font()
        font.setBold(True)
        self.P2_tableWidget_2.horizontalHeader().setFont(font)
        # Import data to GUI
        for i in range (len(data)):
            for j in range (len(data[0])):
                self.P2_tableWidget_2.setItem(i,j,QTableWidgetItem(str(data[i][j])))                 
    def PhD_lab(self):    
    # Which LAB does not have any Master (only PhD) // NOT EXISTS 
        self.tableWidget.clear()
        command_Increase_Salary_PhD = """SELECT L_No
                                        FROM LAB
                                        WHERE NOT EXISTS (SELECT * 
                                                          FROM Student
                                                          WHERE Degree = 'Master' AND LAB_No = L_No)""" 
        cursor.execute(command_Increase_Salary_PhD)
        data = list(cursor.fetchall())  
        self.P2_tableWidget_3.setRowCount(len(data))
        self.P2_tableWidget_3.setColumnCount(len(data[0]))
        self.P2_tableWidget_3.setHorizontalHeaderLabels(["LAB_No."]) 
        font = self.P2_tableWidget_2.horizontalHeader().font()
        font.setBold(True)
        self.P2_tableWidget_3.horizontalHeader().setFont(font)
        # Import data to GUI
        for i in range (len(data)):
            for j in range (len(data[0])):
                self.P2_tableWidget_3.setItem(i,j,QTableWidgetItem(str(data[i][j])))                 
    def Project_student(self):   
        # The project is finished by the student // NOT IN, LIKE              
        self.tableWidget.clear()
        command_project_student = """SELECT ID_Student
                                    FROM Execute
                                    WHERE Project_No NOT IN (SELECT P_No 
                                                             FROM MOST_Project
                                                             WHERE End_Date LIKE '__200731' OR End_Date LIKE '__210731' OR End_Date LIKE '__220731')"""                          
        cursor.execute(command_project_student)
        data = list(cursor.fetchall())  
        self.P2_tableWidget_4.setRowCount(len(data))
        self.P2_tableWidget_4.setColumnCount(len(data[0]))
        self.P2_tableWidget_4.setHorizontalHeaderLabels(["ID_Student"]) 
        font = self.P2_tableWidget_4.horizontalHeader().font()
        font.setBold(True)
        self.P2_tableWidget_4.horizontalHeader().setFont(font)
        for i in range (len(data)):
            for j in range (len(data[0])):
                self.P2_tableWidget_4.setItem(i,j,QTableWidgetItem(str(data[i][j])))               
    def Project_funding(self):
    # The sum of the funding of MOST_project for the Lab // EXISTS, SUM 
        self.tableWidget.clear()
        e_no_2 = str(self.P2_No.text())
        command_Funding = 'SELECT L_No, ID_T, SUM(Funding) FROM LAB, Teacher, MOST_Project WHERE EXISTS (SELECT * FROM Execute'\
            ' WHERE ID_Teacher = ID_T AND LAB_No = L_No AND LAB_No = \'' + e_no_2 + '\' AND P_No = Project_No)'                        
        cursor.execute(command_Funding)
        data = list(cursor.fetchall())  
        self.P2_tableWidget_5.setRowCount(len(data))
        self.P2_tableWidget_5.setColumnCount(len(data[0]))
        self.P2_tableWidget_5.setHorizontalHeaderLabels(["LAB_No.","ID_Teacher","Funding"]) 
        font = self.P2_tableWidget_5.horizontalHeader().font()
        font.setBold(True)
        self.P2_tableWidget_5.horizontalHeader().setFont(font)
        for i in range (len(data)):
            for j in range (len(data[0])):
                self.P2_tableWidget_5.setItem(i,j,QTableWidgetItem(str(data[i][j]))) 
        self.P2_No.clear()
    def Project_maxmin(self):
    # The max & min cost of the equipment in the LAB // MAX, MIN, ORDER, UNION    
        command_Equip = """ ((SELECT LAB_No, E_No, MAX(Cost)
                            FROM Equipment
                            WHERE Cost IN (SELECT MAX(Cost)              
                            FROM Equipment, LAB
                            WHERE LAB_No = L_No
                            GROUP BY LAB_No)
                            GROUP BY LAB_No)
                            UNION
                            (SELECT LAB_No, E_No, MIN(Cost)
                            FROM Equipment
                            WHERE Cost IN (SELECT MIN(Cost)              
                            FROM Equipment, LAB
                            WHERE LAB_No = L_No
                            GROUP BY LAB_No)
                            GROUP BY LAB_No)
                            ORDER BY LAB_No)"""                                                
        cursor.execute(command_Equip)
        data = list(cursor.fetchall())  
        self.P2_tableWidget_6.setRowCount(len(data))
        self.P2_tableWidget_6.setColumnCount(len(data[0]))
        self.P2_tableWidget_6.setHorizontalHeaderLabels(["LAB_No.","ID_Equipment","Cost"]) 
        font = self.P2_tableWidget_6.horizontalHeader().font()
        font.setBold(True)
        self.P2_tableWidget_6.horizontalHeader().setFont(font)
        for i in range (len(data)):
            for j in range (len(data[0])):
                self.P2_tableWidget_6.setItem(i,j,QTableWidgetItem(str(data[i][j])))         
    def Project_avg(self):
    # The avaerage cost of the equipment in the LAB // HAVING, COUNT, AVG, UNION, ORDER
        command_AVG_Equip = """ ((SELECT LAB_No, E_No, Cost
                                FROM Equipment, LAB
                                WHERE LAB_No = L_No
                                GROUP BY LAB_No
                                HAVING COUNT(*) = 1)
                                UNION
                                (SELECT LAB_No, E_No, AVG(Cost)
                                FROM Equipment, LAB
                                WHERE LAB_No = L_No
                                GROUP BY LAB_No
                                HAVING COUNT(*) > 1)
                                ORDER BY LAB_No)"""                                             
        cursor.execute(command_AVG_Equip)
        data = list(cursor.fetchall())  
        self.P2_tableWidget_6.setRowCount(len(data))
        self.P2_tableWidget_6.setColumnCount(len(data[0]))
        self.P2_tableWidget_6.setHorizontalHeaderLabels(["LAB_No.","ID_Equipment","Cost"]) 
        font = self.P2_tableWidget_6.horizontalHeader().font()
        font.setBold(True)
        self.P2_tableWidget_6.horizontalHeader().setFont(font)
        for i in range (len(data)):
            for j in range (len(data[0])):
                self.P2_tableWidget_6.setItem(i,j,QTableWidgetItem(str(data[i][j])))         
    def Data_coding(self):
    # coding for searching the data ex. the data of the LAB
        command_coding = str(self.P4_lineEdit.text())
        cursor.execute(command_coding)        
        data = list(cursor.fetchall())  
        if len(data) == 0 :
            db.commit() 
        else:
            self.P4_tableWidget.setRowCount(len(data))
            self.P4_tableWidget.setColumnCount(len(data[0]))
            self.P4_tableWidget.setHorizontalHeaderLabels(["LAB_No.","Name","Property"]) 
            font = self.P4_tableWidget.horizontalHeader().font()
            font.setBold(True)
            self.P4_tableWidget.horizontalHeader().setFont(font)
            for i in range (len(data)):
                for j in range (len(data[0])):
                    self.P4_tableWidget.setItem(i,j,QTableWidgetItem(str(data[i][j])))     
        self.P4_lineEdit.clear()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainUi()
    window.show()
    app.exec_()
    
    