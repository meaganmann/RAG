# !pip install sentence-transformers
# !pip install faiss-cpu
# !pip install transformers accelerate

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

kb_text = """
Queen's University at Kingston,[3][12][13] commonly known as Queen's University or simply Queen's, is a public research university in Kingston,
Ontario, Canada. Queen's holds more than 1,400 hectares (3,500 acres) of land throughout Ontario and owns Herstmonceux Castle in East Sussex,
England.[9] Queen's is organized into eight faculties and schools.

The Church of Scotland established Queen's College in October 1841 via a royal charter from Queen Victoria. The first classes, intended to prepare
students for the ministry, were held 7 March 1842, with 15 students and two professors.[14] In 1869, Queen's was the first Canadian university west
of the Maritime provinces to admit women.[3] In 1883, a women's college for medical education affiliated with Queen's University was established after
 male staff and students reacted with hostility to the admission of women to the university's medical classes.[15][16] In 1912, Queen's ended its
 affiliation with the Presbyterian Church,[12] and adopted its present name.[17][3] During the mid-20th century, the university established several
 faculties and schools and expanded its campus with the construction of new facilities.

Queen's is a co-educational university with more than 33,842 students and over 131,000 alumni living worldwide.[7][18] Notable alumni include
government officials, academics, business leaders and 62 Rhodes Scholars.[19] As of 2022, five Nobel Laureates and one Turing Award winner have
been affiliated with the university.

The university funds several magazines and journals, among which are the Queen's Quarterly that has been published since 1893.
"""

# Manual chunking — treat each paragraph as a chunk
def chunking(text):
    """
    Manually chunk a text document into smaller pieces.
    This function avoids one-liners and shows each step clearly.
    """
    # Split the text into raw lines
    lines = text.split("\n")

    # Prepare an empty list for chunks
    chunks = []

    # Loop through each line
    for line in lines:

        # Remove leading/trailing whitespace
        cleaned_line = line.strip()

        # Skip empty lines
        if cleaned_line == "":
            continue

        # Add cleaned line as a chunk
        chunks.append(cleaned_line)

    # Return the final list of chunks
    return chunks

chunks = chunking(kb_text)

print("Knowledge Base Chunks:\n")
for i, c in enumerate(chunks):
    print(f"[{i}] {c}\n")

from sentence_transformers import SentenceTransformer
import numpy as np

# Choose a small free embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings for each chunk
chunk_vectors = embedder.encode(chunks, convert_to_numpy=True)

print("Chunk vectors shape:", chunk_vectors.shape)

def cosine_sim(a, b):
    dot_product = np.dot(a, b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    sim = dot_product / (norm_a * norm_b)
    return sim

def retrieve(query, top_k=2):
    '''
    Retrieve the top_k most relevant chunks for the input query
    using manual cosine similarity
    '''

    # Convert the query text into an embedding vector
    q_vec = embedder.encode([query], convert_to_numpy=True)[0]

    # Compute cosine similarity with every chunk vector
    similarities = [cosine_sim(q_vec, chunk_vec) for chunk_vec in chunk_vectors]

    # Find indices of the top_k most similar chunks
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    # Return a list of tuples:
    results = [(i, similarities[i], chunks[i]) for i in top_k_indices]
    return results


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)



def raw_generate(query, top_k=2, max_new_tokens=128, return_full=False):

    # simple direct prompt
    prompt = f"""
                You are a helpful assistant. Answer the following question:

                Question: {query}
                Answer:
            """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def rag_generate(query, top_k=2, max_new_tokens=200, return_full=False):

    # 1. Manual Retrieval
    # Retrieve the top_k most relevant chunks for the input query using manual cosine similarity.
    retrieved = retrieve(query, top_k=top_k)

    retrieved_text = "\n".join([f"- {txt}" for _, _, txt in retrieved])


    # 2. Construct Prompt （change this to fill in retrieved text）
    prompt = f"""<|system|>
    You are a helpful assistant.
    <|user|>
    Use the following context to answer the question.

    Context:
    {retrieved_text}

    Question:
    {query}
    <|assistant|>
    """


    # 3. Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
        temperature=0.7
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


query = "How many students are at Queen's University?"

print("=== Retrieved Chunks ===\n")

for idx, sim, txt in retrieve(query):
    print(f"[{idx}] (score={sim:.3f})\n{txt}\n")

print("\n=== RAG Answer ===\n")
## RAG answer here
rag_answer = rag_generate(query)
print(rag_answer)

print("======================================")
print("            RAW LLM Output")
## RAW LLM Output here
raw_answer = raw_generate(query)
print(raw_answer)
