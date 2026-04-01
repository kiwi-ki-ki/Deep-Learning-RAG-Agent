# System Architecture
## Team: Kiera Wingo
## Date: 03/24/26
## Members and Roles:
- Solo project, all roles completed by Kiera Wingo

---

## Architecture Diagram

Replace this section with your team's completed flow chart.
Export from FigJam, Miro, or draw.io and embed as an image,
or describe the architecture as an ASCII diagram.

The diagram must show:
- [ ] How a corpus file becomes a chunk
- [ ] How a chunk becomes an embedding
- [ ] How duplicate detection fires
- [ ] How a user query flows through LangGraph to a response
- [ ] Where the hallucination guard sits in the graph
- [ ] How conversation memory is maintained across turns

*(replace this line with your diagram image or ASCII art)*

---

## Component Descriptions

### Corpus Layer

- **Source files location:** `data/corpus/`
- **File formats used:** .md initially, with optional support for .pdf later

- **Landmark papers ingested:**
  *(list the papers your team located and ingested, one per line)*
  - 
  -
  -

- **Chunking strategy:**
 The initial strategy is to split documents into small semantic chunks that each contain one main idea. A practical target is about 300 to 500 characters with light overlap, or one subsection per chunk if the markdown headings are already well structured. This was chosen to keep chunks specific enough for retrieval while still containing enough explanation for interview-style questions.

- **Metadata schema:**
  *(list every metadata field your chunks carry and explain why each field exists)*
  | Field | Type | Purpose |
  |---|---|---|
  | topic | string | identifies the main deep learning topic of the chunk |
  | difficulty | string | supports question generation and possible filtering |
  | type | string | identifies whether the chunk is a concept, comparison, paper summary, or other content type |
  | source | string | tracks the file the chunk came from for citation |
  | related_topics | list | connects topics such as RNN and LSTM for cross-topic retrieval |
  | is_bonus | bool | marks optional topics outside the core required scope |

- **Duplicate detection approach:**
  Duplicate detection is based on chunk content rather than filename alone. A content hash or deterministic chunk ID is more reliable than a filename because different files can contain identical text, and the same file can be renamed without changing its content.

- **Corpus coverage:**
  - [X] ANN
  - [X] CNN
  - [X] RNN
  - [ ] LSTM
  - [ ] Seq2Seq
  - [ ] Autoencoder
  - [ ] SOM *(bonus)*
  - [ ] Boltzmann Machine *(bonus)*
  - [ ] GAN *(bonus)*

---

### Vector Store Layer

- **Database:** ChromaDB — PersistentClient
- **Local persistence path:** data/chroma_db

- **Embedding model:**
  sentence-transformers/all-MiniLM-L6-v2 via Hugging Face sentence-transformers

- **Why this embedding model:**
  This model was chosen because it is lightweight, fast, widely used, and works well on a local machine without requiring paid API calls for embeddings. It is a strong balance between speed and semantic quality.

- **Similarity metric:**
  Cosine similarity, because it works well for comparing semantic embeddings and is a common default for text retrieval tasks.

- **Retrieval k:**
  k = 3 to 5, with 4 as a practical default. This is enough to give the model useful context without overwhelming the prompt with too much unrelated material.

- **Similarity threshold:**
  A minimum similarity threshold is used before generation. If the retrieved chunks do not meet the threshold, the system returns a no-context response instead of forcing an answer. The exact threshold may need small manual tuning during testing.

- **Metadata filtering:**
  The design allows future filtering by topic or difficulty using chunk metadata. In the first version, retrieval is primarily semantic, but metadata fields are included so filtering can be added easily later.

---

### Agent Layer

- **Framework:** LangGraph

- **Graph nodes:**
  *(describe what each node does in one sentence)*
  | Node | Responsibility |
  |---|---|
  | query_rewrite_node | rewrites the user query into a cleaner retrieval-focused version |
  | retrieval_node | queries the vector store and returns the top matching chunks |
  | generation_node | generates the final answer using only retrieved context and adds citations |

- **Conditional edges:**
  If retrieval returns no strong matches or scores below the similarity threshold, the graph routes to a guard response instead of generation. If relevant context is found, the graph continues to the generation node.

- **Hallucination guard:**
  The system returns a message such as:
  "No relevant context was found in the current corpus, so I cannot answer this reliably."

- **Query rewriting:**
  *(give one example of a raw user query and how your system rewrites it)*
  - Raw query: how do lstms help compared to rnns
  - Rewritten query: Explain how LSTMs improve on standard RNNs, especially for long-term dependencies

- **Conversation memory:**
  Conversation history is maintained in session state so that user questions and assistant responses persist across turns during the active app session. If the session resets or the app restarts, the memory is lost unless persistence is added later.

- **LLM provider:**
  Planned provider: Groq
  Planned model: llama-3.1-8b-instant

- **Why this provider:**
  Groq was selected because it is fast, simple to set up, and does not require local GPU resources. It is a practical choice for getting a working demo running on a Mac.

---

### Prompt Layer

- **System prompt summary:**
  The system prompt defines the agent as a deep learning interview preparation assistant that must answer only from retrieved context, cite sources, and avoid unsupported claims. Its core constraint is that it should not invent information outside the corpus.

- **Question generation prompt:**
  This prompt takes a retrieved chunk and a difficulty level as input and returns a structured interview question, a model answer, a follow-up question, and source citations.

- **Answer evaluation prompt:**
  This prompt takes a question, a candidate answer, and the relevant context, then scores the answer on a 1 to 10 scale with feedback about missing details, accuracy, and completeness.

- **JSON reliability:**
  To improve reliability, the prompts explicitly instruct the model to return only a JSON object with no prose, no markdown fences, and no extra explanation.

- **Failure modes identified:**
  *(list at least one failure mode per prompt and how you addressed it)*
  - The system prompt may still allow the model to add outside knowledge if the constraint is too weak
  - The question generation prompt may return malformed JSON or a weak question
  - The answer evaluation prompt may score too generously unless the rubric is specific

---

### Interface Layer

- **Framework:** Streamlit
- **Deployment platform:** HuggingFace Spaces or Streamlit Community Cloud
- **Public URL:** *(paste your deployed app URL here once live)*

- **Ingestion panel features:**
  The ingestion panel is designed to allow file upload, show upload status, and display the list of ingested documents.

- **Document viewer features:**
  The document viewer allows users to browse ingested files and inspect chunk content so they can see what material is available to the agent.

- **Chat panel features:**
  The chat panel displays the conversation history, accepts user questions, shows source citations with each answer, and surfaces the hallucination guard when no relevant context is found.

- **Session state keys:**
  *(list the st.session_state keys your app uses and what each stores)*
  | Key | Stores |
  |---|---|
  | chat_history | prior user and assistant messages |
  | ingested_documents | the files currently loaded into the application |
  | selected_document | the document currently being viewed |
  | thread_id | the current conversation or graph thread identifier |

- **Stretch features implemented:**
  None in the first version. The focus is on a working ingestion, retrieval, and grounded response pipeline.

---

## Design Decisions

Document at least three deliberate decisions your team made.
These are your Hour 3 interview talking points — be specific.
"We used the default settings" is not a design decision.

1. **Decision:**
   Documents are split into semantically meaningful chunks based on sections or paragraphs. 
   Each chunk is approximately 300 to 600 words in length with a small overlap of about 
   50 to 100 characters between adjacent chunks. This overlap helps preserve context 
   across boundaries while still keeping chunks focused on a single idea.
   **Rationale:**
   Markdown is cleaner, easier to chunk, and faster to debug than PDFs, which often introduce noisy text extraction. Starting with markdown reduces ingestion problems and makes retrieval easier to validate.
   **Interview answer:**
   I prioritized clean markdown ingestion first because retrieval quality depends heavily on chunk quality. PDFs were left as an extension because extraction noise would have made debugging much harder in an early prototype.

2. **Decision:** Use a lightweight local embedding model from Hugging Face
   **Rationale:** A small sentence-transformer model provides good semantic retrieval while remaining practical on a Mac. This choice avoids embedding API cost and keeps the pipeline reproducible.
   **Interview answer:** I chose all-MiniLM-L6-v2 because it gives a good speed-quality tradeoff for semantic search. It is fast enough to run locally while still producing embeddings strong enough for a small educational corpus.

3. **Decision:** Add a hallucination guard before generation
   **Rationale:** If retrieval is weak, generating an answer anyway would reduce trust in the system. A guard ensures the system fails safely by admitting when it lacks relevant context.
   **Interview answer:** We treated retrieval confidence as a gate before answer generation. If the similarity signal is too weak, the system returns a no-context message instead of hallucinating, which makes the agent more trustworthy.

4. **Decision:** Keep conversation memory in session state for the first version
   **Rationale:** Session state is the simplest way to preserve chat history in Streamlit without adding database complexity. This is enough for a demo and can later be replaced with persistent memory if needed.
   **Interview answer:** I used session state as a lightweight memory layer because it was simple and worked well for a single-session prototype. It preserves multi-turn interaction without complicating the architecture with extra storage.

---

## QA Test Results

*(QA Lead fills this in during Phase 2 of Hour 2)*

| Test | Expected | Actual | Pass / Fail |
|---|---|---|---|
| Normal query | Relevant chunks, source cited |  | |
| Off-topic query | No context found message |  | |
| Duplicate ingestion | Second upload skipped |  | |
| Empty query | Graceful error, no crash |  | |
| Cross-topic query | Multi-topic retrieval |  | |

**Critical failures fixed before Hour 3:**
-
-

**Known issues not fixed (and why):**
-
-

---

## Known Limitations

Be honest. Interviewers respect candidates who understand
the boundaries of their own system.

- The initial corpus is small, so coverage of deep learning topics is incomplete
- Similarity threshold tuning is manual rather than empirically calibrated
- Conversation memory is session-based and resets when the app restarts
- Early PDF ingestion may produce noisy chunks depending on extraction quality

---

## What We Would Do With More Time

- Add hybrid retrieval with both semantic and keyword search
- Add a re-ranking step for better chunk ordering
- Improve PDF parsing and chunk cleaning
- Expand corpus coverage to LSTM, Seq2Seq, autoencoders, GANs, and landmark papers
- Add persistent conversation memory across sessions

---

## Hour 3 Interview Questions

*(QA Lead fills this in — these are the questions your team
will ask the opposing team during judging)*

**Question 1:**

Model answer:

**Question 2:**

Model answer:

**Question 3:**

Model answer:

---

## Team Retrospective

*(fill in after Hour 3)*

**What clicked:**
-

**What confused us:**
-

**One thing each team member would study before a real interview:**
- Corpus Architect:
- Pipeline Engineer:
- UX Lead:
- Prompt Engineer:
- QA Lead:
