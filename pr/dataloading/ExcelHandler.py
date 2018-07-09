

class ExcelHandler:
@staticmethod
def loadFromFile():





df = pd.read_excel('Arizona State.xlsx')
# df['all'] = df.apply( ''.join, axis=1)
df['Summary'] = pd.Series(df.fillna('  ').values.tolist()).str.join(' ')
df['Summary'] = df['Summary'].apply(lambda x: PreProcessing.filterStopWords(x, customStopWords = ["*","•","arizona","http","llc","aztecom","this","This"])).str.join(' ')
writer = pd.ExcelWriter('output.xlsx')
df.to_excel(writer,'Sheet1')
doc_complete = df["Summary"].as_matrix()
