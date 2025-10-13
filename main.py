

"""

    i want an cli if else for this 
    i want to choose what to do 

    first :
    in mydocuments folder
    ill have pdf documents 
    those documents will be converted to chunks.jsonl using the ingest_own_docs.py script
    then using build_pinecone.py script, i'll put the chunks.jsonl to pinecone

    second:
    in the data folder i'll have passages.jsonl 
    which will be indexed using build_bm25.py script 


    common for both first and second:
    printing format:
        for each query 
        the similarity score of that retrived chunk 
        the relevant chunks 
    
    common for both first and second:
    then in the end 
    hit gemini via ask.py script

    

"""h
