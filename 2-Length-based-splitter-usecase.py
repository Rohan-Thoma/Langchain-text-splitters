from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Data/dl-curriculum.pdf")

#load the pdf pages as docs
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator=''
)
 
#Here we are passing the documents directly instead of text
#ALs we change the function from split_text to split_documents()
result = splitter.split_documents(docs)

print("\n=========================\n")
print("Number of chunks = ",len(result))
print("\n=========================\n")
#Since this is a large document, visualizing all the chunks doesnt make sense so
# we will visualize only 1st chunk
print("1st chunk  = ", result[0])
