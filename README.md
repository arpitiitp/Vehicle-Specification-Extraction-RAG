# Predii AI: Service Manual Spec Extractor

A RAG (Retrieval-Augmented Generation) pipeline built inside a Jupyter Notebook that reads a vehicle service manual PDF and lets users extract structured specifications — torque values, fluid capacities, alignment specs, and more — through both a batch extraction job and an interactive GUI.

---

## What This Project Does

Vehicle service manuals are dense PDFs full of tables, part numbers, and technical specs. Finding a specific torque value or fluid capacity manually is slow and error-prone. This project automates that by:

1. Ingesting the PDF and breaking it into searchable chunks
2. Creating a semantic vector index so we can find the right section of the manual for any query
3. Sending those relevant sections to a large language model (LLM) to extract structured data
4. Presenting the results in a clean, interactive table inside the notebook

---

## Architecture Overview

```
PDF → PyMuPDF → Text Chunks → HuggingFace Embeddings → FAISS Index
                                                              ↓
User Query → Similarity Search → Top-k Chunks → Groq Llama 3.3 → JSON Output → Table UI
```

---

## Cell-by-Cell Walkthrough

### Cell 1 — Installing Dependencies

```python
%pip install langchain langchain-community langchain-openai pymupdf faiss-cpu pydantic python-dotenv
%pip install langchain-ollama
%pip install langchain-groq
%pip install sentence-transformers langchain-huggingface
!pip install ipywidgets
```

**Why these packages?**

| Package | Purpose |
|---|---|
| `langchain` / `langchain-community` | Core orchestration framework for chaining LLM calls, retrievers, and prompts |
| `pymupdf` | PDF reader — chosen because it preserves table structure better than alternatives like `pdfplumber` for this specific manual |
| `faiss-cpu` | Facebook's vector similarity library — fast, local, no server needed |
| `sentence-transformers` / `langchain-huggingface` | Local embedding model, zero API cost |
| `langchain-groq` | Groq cloud LLM integration |
| `pydantic` | Data validation and schema definition for output structure |
| `python-dotenv` | Loads API keys from a `.env` file so secrets don't live in the notebook |
| `ipywidgets` | Powers the interactive GUI in the last cell |



---

### Cell 2 — Imports and Configuration

```python
PDF_PATH = "sample-service-manual.pdf"
```

All imports are collected here and the PDF path is set as a single constant. This makes it trivial to swap in a different manual — just change one variable.

---

### Cell 3 — Loading and Inspecting the PDF

```python
loader = PyMuPDFLoader(PDF_PATH)
documents = loader.load()
print(documents[1].page_content[:1000])
```

**Why PyMuPDF?**
PyMuPDF's raw text output is sufficient here.


---

### Cell 4 — Chunking the Document

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=300,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
```


**Why `chunk_size=2000`?**
This was set deliberately large. Service manual tables can span 15–20 rows, each row being one spec. A chunk size of 500 (a common default) would split a table in half, causing the model to miss half the data. 2000 characters is typically enough to hold a full specification table on one page.

**Why `chunk_overlap=300`?**
If a specification happens to fall right at the boundary between two chunks, overlap ensures neither chunk loses critical context. The 300-character overlap means a spec at the end of chunk N also appears at the beginning of chunk N+1 — the retriever can catch it either way.

**Why `RecursiveCharacterTextSplitter`?**
It tries to split on natural boundaries first (`\n\n` for paragraphs, then `\n` for lines, then spaces). This is better than a naive character split that might cut in the middle of a number or a component name.

---

### Cell 5 — Embeddings and Vector Store

```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
```

**Why local embeddings instead of OpenAI?**
The OpenAI embeddings API costs money per token. For a student project processing a large PDF, the cost adds up quickly. `all-MiniLM-L6-v2` is a free, locally-run model (~80MB download) that is specifically designed for semantic similarity tasks. Its accuracy for retrieval is very competitive with paid APIs for technical text.

**Why FAISS?**
FAISS (Facebook AI Similarity Search) is a mature, battle-tested library for fast approximate nearest-neighbor search on dense vectors. It runs entirely in memory without needing a separate database server (unlike Chroma or Pinecone), which makes setup simpler. The index is also saved to disk (`faiss_db_index/`) so it doesn't have to be rebuilt every time the notebook is opened.

**Tradeoff:** FAISS is great for this scale (a single PDF), but it loads everything into RAM. For a library of hundreds of manuals, a server-based solution like Pinecone or Qdrant would be more appropriate.

---

### Cell 6 — Testing Retrieval (Debug Step)

```python
test_query = "What is the correct torque specification for the front tie-rod end jam nut"
results = vector_store.similarity_search(test_query, k=10)
```

This cell exists purely to verify the retrieval is working before involving the LLM. By printing the top results, we can confirm:
- The correct pages are being retrieved
- The chunks contain the actual table data we need
- `k=10` is used here (more than the final pipeline's `k=3`) to get a broad view of what the index is surfacing


---

### Cell 7 — Output Schema (Pydantic Models)

```python
class VehicleSpec(BaseModel):
    component: str
    spec_type: str
    value: str
    unit: Optional[str]

class SpecList(BaseModel):
    specs: List[VehicleSpec]
```

These Pydantic models formally define what a valid extracted specification looks like. They serve as:

1. **Documentation** — anyone reading the code immediately understands the output shape
2. **A schema reference** — the prompt instructs the LLM to produce JSON that matches this structure

**Note:** These models are not wired into the LLM call via `.with_structured_output()`. Instead, the JSON is extracted manually using regex (see Cell 8). The reason: Groq's structured output enforcement was not reliable for all models at the time of development, and the manual regex approach gave more control over fallback behavior.

---

### Cell 8 — Main Extraction Loop (The Core Logic)

This is the most important cell. It has four distinct parts:

#### Part 1: LLM Setup

```python
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)
```

**Why Groq?**
Groq provides a free API tier for running Llama 3.3 70B. The original design used Ollama (fully local), but running a 70B model locally requires significant hardware. Groq's inference hardware (LPU chips) is also extremely fast — responses come back in 1–3 seconds, which is important for the interactive GUI.

**Why `llama-3.3-70b-versatile`?**
This is one of the best open-weight models available. For structured data extraction from technical tables, a larger, more capable model reduces the chance of the model hallucinating values or misassigning units.

**Why `temperature=0`?**
Extraction is not a creative task. We want the model to be deterministic and literal — output exactly what it reads in the context. Temperature 0 eliminates randomness.

#### Part 2: The JSON Extractor Helper

```python
def extract_json_from_text(text):
    text = text.replace("```json", "").replace("```", "")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(text)
```

**Why is this needed?**
LLMs often wrap their JSON output in conversational text like *"Sure! Here is the extracted data:"* followed by the JSON. Even with a strict prompt, this happens occasionally. Using `json.loads()` directly on the full response would crash. This helper strips markdown code fences, then uses a regex to find the first `{` and last `}` in the response — reliably extracting just the JSON object regardless of any surrounding text.

This is a practical, battle-tested pattern for working with LLM output in production.

####  Batch Run and Save

```python
queries = [
    "Torque specifications for front suspension",
    "Torque specifications for braking system",
    "Fluid capacities"
]
```

Three preset queries are run automatically to extract the most commonly needed spec categories. Results are collected into `all_extracted_data` and saved to `vehicle_specs.json`. This JSON file acts as a persistent cache — it doesn't need to be regenerated every time.

---

### Cell 9 — Save and View Results

```python
with open(output_file, "w") as f:
    json.dump(all_extracted_data, f, indent=4)

print(json.dumps(all_extracted_data[:5], indent=2))

vector_store.save_local("faiss_db_index")
```

The FAISS index is saved to disk here as `faiss_db_index/index.faiss` and `faiss_db_index/index.pkl`. This means on subsequent runs, the index can be loaded directly — skipping the embedding computation entirely (which takes ~30–60 seconds on first run).

---

### Last Cell — Interactive GUI (Predii AI: Service Hub)

This is the user-facing part of the application. It uses `ipywidgets` to build a complete interactive interface inside the notebook.

#### What the user sees:

```
┌─────────────────────────────────────────────┐
│  Predii AI: Service Hub                     │
│  Powered by Groq Llama 3.3                  │
├─────────────────────────────────────────────┤
│  Quick Select: [Dropdown ▼]                 │
│  Custom Query: [__________________________] │
│  [ Extract Data 🔍 ]                        │
├─────────────────────────────────────────────┤
│  Component    | Type   | Value | Unit       │
│  Tie-Rod Nut  | Torque | 17    | Nm         │
│  ...                                        │
└─────────────────────────────────────────────┘
```

#### How it works:

**Dropdown for Quick Queries:**
```python
PRESET_QUERIES = [
    "Torque specifications for front suspension",
    "Torque specifications for braking system",
    "Service materials",
    "Wheel alignment specifications",
    "Grease and lubricants"
]
```
A pre-populated dropdown gives users one-click access to the most common query types. When a preset is selected, it auto-fills the text input field below it.

**Free-text Input:**
Users can also type any custom question into the text box — the system is not limited to the preset queries. This makes it a general-purpose service manual assistant.

**Search Button (`on_search_click`):**
When clicked, the button triggers the full RAG pipeline live:
1. Runs `vector_store.similarity_search(q, k=3)` to find relevant manual sections
2. Builds the prompt with those sections as context
3. Calls Groq Llama 3.3 and waits for a response
4. Parses the JSON using `clean_and_parse_json` (same robust helper as Cell 8)
5. Renders an HTML table with the results, or shows an error message if nothing was found

**Styled Output Table:**
Results are rendered as a formatted HTML table with custom CSS — highlighted value column in orange, hover effects on rows, and a branded header. This was done because the default `print()` output in Jupyter is hard to read for tabular data.

**Error Handling:**
- Empty query → shows a warning
- Valid JSON with no specs → shows "No data found" message
- Network/API error → catches the exception and prints it cleanly

---


## Setup and Running

### Prerequisites
- Python 3.9+
- Anaconda or any Jupyter environment
- A free [Groq API key](https://console.groq.com/)

### Step 1: Create your `.env` file

```
GROQ_API_KEY=your_groq_api_key_here
```

### Step 2: Install dependencies

Run Cell 1 of the notebook. All packages will be installed automatically.

### Step 3: Run the notebook top to bottom

- **Cells 1–5**: One-time setup. Builds and saves the FAISS index. Takes ~1–2 minutes on the first run (downloading the embedding model).
- **Cell 6**: Optional debug step — verify retrieval is working.
- **Cell 8**: Batch extraction — generates `vehicle_specs.json` with pre-defined queries.
- **Last cell**: Launch the interactive GUI.

> On subsequent runs, if you only want the GUI, you can skip Cell 5 and instead load the saved index:
> ```python
> vector_store = FAISS.load_local("faiss_db_index", embeddings, allow_dangerous_deserialization=True)
> ```

---


## Limitations

- **PDF quality matters:** The pipeline depends on PyMuPDF being able to read the PDF text layer. Scanned/image-only PDFs will not work without adding an OCR step.
- **Table extraction is heuristic:** PyMuPDF reads text sequentially. Complex multi-column layouts may be read in the wrong order, causing the LLM to misinterpret values.
- **k=3 may miss specs:** If the relevant data is spread across many pages, only the 3 closest chunks are sent to the model. Some rows in large tables could be missed.
- **Groq rate limits:** The free tier has per-minute token limits. Running many queries in quick succession may trigger throttling.

---

## Possible Improvements

- Load the saved FAISS index on startup instead of rebuilding it every run
- Use Re-Ranking for Improving Retrival Relevency/Accuracy.
- Add a CSV export button to the GUI
- Increase `k` to 5–6 for more complete table coverage, with a token budget check
- Add an OCR fallback for scanned PDFs using `pytesseract`
- Replace the Pydantic models as documentation-only with actual `.with_structured_output()` enforcement once Groq's support stabilizes
