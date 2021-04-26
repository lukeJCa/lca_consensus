# Use Tkinter for python 2, tkinter for python 3
import tkinter as tk
from tkinter import Menu, Frame
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
import pandas as pd
from pandastable import Table, TableModel
from classification import linear_model, svm_model, random_forest_model, mlp_model

class MainApplication(tk.Frame):
	def __init__(self, parent, *args, **kwargs):
		tk.Frame.__init__(self, parent, *args, **kwargs)
		self.parent = parent
		root = self.parent
		root.title("Graphical Model Comparison Tool - Python3")
		root.geometry('600x400+200+100')
		root.iconphoto(False, tk.PhotoImage(file='brain.png'))
		menubar = Menu(root)
		filemenu = Menu(menubar, tearoff=0)
		filemenu.add_command(label="Open", command=self.load_file)
		filemenu.add_command(label="Close", command=self.close_data)

		filemenu.add_separator()

		filemenu.add_command(label="Exit", command=root.quit)
		menubar.add_cascade(label="Data", menu=filemenu)
		editmenu = Menu(menubar, tearoff=0)
		editmenu.add_command(label="Select Desired Column", command=self.get_labels)

		editmenu.add_separator()

		editmenu.add_command(label="Linear Regression", command=self.linear_regression)
		editmenu.add_command(label="SVM", command=self.support_vector_machine)
		editmenu.add_command(label="Random Forest", command=self.random_forest)
		editmenu.add_command(label="Multi Layer Perceptron", command=self.multilayer_perceptron)

		menubar.add_cascade(label="Model", menu=editmenu)
		root.config(menu=menubar)

	def multilayer_perceptron(self):
		mlp = mlp_model.MLPModel(self.df, self.label)
		print(mlp.get_score())
		
	def random_forest(self):
		rf = random_forest_model.RandomForestModel(self.df, self.label)
		print(rf.get_score())

	def support_vector_machine(self):
		svm = svm_model. SVCModel(self.df, self.label)
		print(svm.get_score())

	def linear_regression(self):
		linear = linear_model.LinearModel(self.df, self.label)
		print(linear.get_score())

	def get_labels(self):
		## Returns the column number current selected
		column_no = int(self.table.getSelectedColumn())
		self.label = self.df.columns[column_no]

	def close_data(self):
		self.f.pack_forget()

	def load_file(self):
		self.df = []
		fname = askopenfilename(title = "Select file", filetypes=(("CSV File", "*.csv"),))
		if fname:
			try:
				self.df = pd.read_csv(fname)
			except:                     # <- naked except is a bad idea
				showerror("Open Source File", "Failed to read file\n'%s'" % fname)

		self.f = Frame(self.parent)
		self.f.pack(fill='both',expand=1)
		self.table = pt = Table(self.f, dataframe=self.df,showtoolbar=True, showstatusbar=True)
		pt.show()

	def donothing(self):
		print(" A button")
if __name__ == "__main__":
	root = tk.Tk()
	MainApplication(root).pack(side="top", fill="both", expand=True)
	root.mainloop()