# Node names in the graph
RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
WEB_SEARCH = "web_search"
GENERATE = "generate"

# Number of documents the retriever returns per question
RETRIEVER_K = 5

# Fewer relevant documents than this after grading -> fall back to web search
MIN_RELEVANT_DOCUMENTS = 3

# Number of results a single Tavily web search returns
MAX_SEARCH_RESULTS = 3

# Outcomes of grading a generation
USEFUL = "useful"
NOT_USEFUL = "not useful"
NOT_SUPPORTED = "not supported"
