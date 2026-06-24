from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader

docs = []

# 1) PDF
# pdf_loader = PyPDFLoader("docs/ai_intro.pdf")
# docs.extend(pdf_loader.load())

# 2) Word
word_loader = Docx2txtLoader("../../data/ai_notes.docx")
docs.extend(word_loader.load())

# 3) Excel (turns sheets into text)
# excel_loader = UnstructuredExcelLoader("docs/ai_data.xlsx")
# docs.extend(excel_loader.load())

print(f"Loaded {len(docs)} documents/chunks")
for d in docs[:3]:
    print("----")
    print(d.page_content[:200])
