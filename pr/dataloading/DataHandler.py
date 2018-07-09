import pandas as pd
import numpy as np
import os
from pathlib import Path

class FilesHandler:
	@staticmethod
	def ifFileExists(filePath):
		my_file = Path(filePath)
		return my_file.is_file()




class ExcelHandler:
	@staticmethod
	def convertToDataframe(filePath):
		df = pd.read_excel(filePath)
		head, tail = os.path.split(filePath)
		print(tail)
		df["fname"] = tail;
		return df


	@staticmethod
	def loadFromDirectory(DirectoryPath, format="", isDebug = False, specificFileName=""):
		all_data = pd.DataFrame()
		for file in os.listdir(DirectoryPath):
    			if file.endswith(format) and isDebug and file.startswith(specificFileName):
    					print(os.path.join(DirectoryPath, file))
		result = [os.path.join(DirectoryPath, file) for file in os.listdir(DirectoryPath) if file.endswith(format)  if file.startswith(specificFileName) ]
		return result

	@staticmethod
	def loadIntoDataframe(dataFiles):
		all_data = pd.DataFrame()
		for file in dataFiles:
			df = ExcelHandler.convertToDataframe(file)
			all_data = all_data.append(df,ignore_index=True)
			# print("Total rows: {0}".format(len(df)))
		return all_data
	
	@staticmethod
	def writeToFile(df, fileFullPath):
		writer = pd.ExcelWriter(fileFullPath)
		df.to_excel(writer,'Sheet1')
	
	@staticmethod
	def describe(df):
		print("Total rows: {0}".format(len(df)))







# f = ExcelHandler.convertToDataframe("D:\\Work\\PR\\Seventh AI\\Code\\Dataset\\test.xlsx")
# print(f)
# ExcelHandler.writeToFile(f,"D:\Work\PR\Seventh AI\Code\Dataset\\t.xlsx")