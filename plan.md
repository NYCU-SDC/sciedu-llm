# Expand sciedu-llm with an LLM-as-a-Judge Evaluation Service

## Context

`sciedu-llm` is a FastAPI service that proxies chat to a self-hosted OpenAI-compatible endpoint (NCHC). It has a single `/chat` SSE route, no RAG, no Langfuse wiring, and no eval. The lab needs a reproducible benchmark over its 1,740 secondary-school science questions (`data/{biology,chemical,physical}_questions.xlsx`) so it can compare LLMs and prompt revisions over time.

We'll add a shared RAG pipeline (BM25 + bge-m3 dense + RRF + BGE-Reranker-V2-M3) and a separate Gradio service that runs the eval, traces every step to Langfuse, and uploads scores. Datasets and prompts live on Langfuse so non-engineers can edit them.

User-locked decisions:
- Embeddings: `bge-m3` via existing `OPENAI_BASE_URL`.
- Reranker: `BGE-Reranker-V2-M3` via an OpenAI-compatible `/rerank` endpoint.
- Judge: separate, configurable model.
- Entry point: a Gradio frontend, separate from the FastAPI app.

## Implementation plan 

### Model Definitions

Langfuse Dataset:
- Provided within the Langfuse sdk
- Has name as string
- Each entry has an item_id and attrs input, output and metadata, each a dictionary

Corpus Dataset (Langfuse Dataset)
- One dataset contains one subject-grade (e.g., physics grade 9), each entry is a chapter
- Input is dict of {content: string}
- Output is empty
- Metadata contains dict of {chapter: string}

Questions Dataset (Langfuse Dataset)
- Input is dict of {question: string}
- Output is dict of {gold_answer: string, ref_text: string[]}
- Metadata is dict of {ref_text_coords: string[]}

Prompts: 
Prompts are special langfuse objects which are retrieved using the langfuse client `get_prompt` function. The following prompts are defined in this system
- `rag-generator-instruction`: The developer prompt for the LLM that generates an answer based on context 
- `judge-factuality`: The developer prompt for the LLM-as-a-judge model which specifically judges factuality
- `judge-conciseness`: The developer prompt for the LLM-as-a-judge model which specifically judges conciseness
- `extract-score-from-judgement`: Instructions to read a output judgement and output a single number in case the LLM-as-a-judge's model fails to provide a score at the end of generation

### 0. Create data seeding system

The current data/ directory has the following structure

```
data/
├── corpus
│   ├── subject
│   │   ├── subject_grade_chX.txt
│   │   ├── ...
│   │   └── subject_grade_chX.txt
│   └── subject
│       ├── subject_grade_chX.txt
│       ├── ...
│       └── subject_grade_chX.txt
├── questions
│   ├── biology_questions.xlsx
│   ├── chemical_questions.xlsx
│   └── physical_questions.xlsx
└── scripts
```

The questions/ folder contain the raw material for the Questions Dataset. Questions are separated by subject, and are provided in xlsx format. Each xlsx has the following important headers and other metadata
- 題目內容: Input dict's question: string
- 答案分析: Ideal answer
- 參考段落: The reference text used to generate the ideal answer, multiple references are separated with a line that only has "---" on it
- 絕對座標: The reference text coordinates in `filename(char_start-char_end)` flavor. Multiple entries are separated via semicolons. For example: `physical_10_ch1.txt(2674-2761); physical_10_ch1.txt(2794-2862); physical_10_ch1.txt(4383-4437)`

The corpus folder contains the actual textbooks for each subject, separated by chapter.

Write seeding scripts in data/scripts such that when run, the corresponding langfuse datasets are automatically created.

### 1. Implement RAG system

The RAG system handles two stages: Construction and Generation. Each stages' behavior is defined as follows

#### Construction (Sync)

This is when the system is initialized with a dataset. The system will be provided with the following:

- A list of corpus datasets
- An embedding model id

Given these, the system should 
1. Merge the corpus datasets into a pandas dataframe
2. Chunk the dataframe based on paragraphs, and record the coordinates of the original text -> chunks (e.g., {textbook: physical_10_ch1, range: (2674, 2761)} -> {chunks: [260, 261, 262]})
3. Provide a helper function which resolves textbook/coordinate pairs to chunk IDs
4. Generate embeddings via the OpenAI API with the bge-m3 model and record in a FAISS database

#### Generation (Async)

This is when a user query comes in. Given the user prompt and a generation model, the system should

1. Generate the prompt's embeddings
2. Run BM25 + Cosine Similarty Search
3. Merge the search results via RFF
4. Run BGE-Reranker-V2-M3 via OpenAI 
5. Fill the query results into the `rag-generator-instruction` prompt and call the OpenAI API
6. Return a dictionary of {question_id: "the ID of this question in the Question Dataset", output_text: string, reference_chunks: chunk_ids[]}

### 2. Implement the judge module

Given a Question Dataset, the judge will rate the RAG system based on two dimensions: Retrival Accuarcy and Generation Quality. The judge module will be given the following information

- A list of Corpus Datasets
- A list of Question Datasets
- A `k` for workflow step 3

### Judge module workflow
1. Import the RAG module and construct the RAG system with the list of corpus datasets
2. Aggregate the question datasets, and run RAG generation for each
3. Convert the question dataset's ref_text_coords back into chunk_ids via the RAG system's helper function, then judge retrival accuracy on recall@k, precision@k, f1@k and MRR
4. Call the OpenAI API with prompts `judge-factuality` and `judge-conciseness` to evaluate model generation quality. Extract the last word (space-separated) from the answer. If the extraction fails, repeatedly call the OpenAI API with prompt `extract-score-from-judgement` until a score is outputed
5. Aggregate all scores and submit to langfuse

### 3. Implement the user facing service

Write a gradio frontend which takes `eval_model_id`, `judge_model_id`, a list of corpus dataset names, a list of question dataset names, and a k, and starts an evaluation with the two given models. Make sure the evaluation continues even after the tab is closed.

### Implementation Reminders
- All API calls to the OpenAI API share a rate limit, so use a semaphore to prevent overwhelming the server.
- The RAG service will later be used in app/ too, so it can't depend on anything in app/
