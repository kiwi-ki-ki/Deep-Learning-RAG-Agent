"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.

    Parameters
    ----------
    settings : Settings, optional
        Application settings.

    Example
    -------
    >>> chunker = DocumentChunker()
    >>> chunks = chunker.chunk_file(
    ...     Path("data/corpus/lstm.md"),
    ...     metadata_overrides={"topic": "LSTM", "difficulty": "intermediate"}
    ... )
    >>> print(f"Produced {len(chunks)} chunks")
    """

    # Default chunking parameters — justify these in your architecture diagram.
    # chunk_size: 512 tokens balances context richness with retrieval precision.
    # chunk_overlap: 50 tokens prevents concepts that span chunk boundaries
    # from being lost entirely. A common interview question.
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """
        Load a file and split it into DocumentChunks.

        Automatically detects file type and routes to the appropriate
        loader. Applies metadata_overrides on top of auto-detected
        metadata where provided.

        Parameters
        ----------
        file_path : Path
            Absolute or relative path to the source file.
        metadata_overrides : dict, optional
            Metadata fields to set or override. Keys must match
            ChunkMetadata field names. Commonly used to set topic
            and difficulty when the file does not encode these.
        chunk_size : int
            Maximum characters per chunk.
        chunk_overlap : int
            Characters of overlap between adjacent chunks.

        Returns
        -------
        list[DocumentChunk]
            Fully prepared chunks with deterministic IDs and metadata.

        Raises
        ------
        ValueError
            If the file type is not supported.
        FileNotFoundError
            If the file does not exist at the given path.
        """
        # TODO: implement
        # 1. Validate file exists
        # 2. Route to _chunk_pdf or _chunk_markdown based on suffix
        # 3. Apply metadata_overrides
        # 4. Generate chunk_ids using VectorStoreManager.generate_chunk_id
        # 5. Return list[DocumentChunk]
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            raw_chunks = self._chunk_pdf(file_path, chunk_size, chunk_overlap)
        elif suffix == ".md":
            raw_chunks = self._chunk_markdown(file_path, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        base_metadata = self._infer_metadata(file_path, metadata_overrides)
        document_chunks: list[DocumentChunk] = []

        for raw_chunk in raw_chunks:
            chunk_text = raw_chunk["text"].strip()
            if not chunk_text:
                continue

            chunk_id = VectorStoreManager.generate_chunk_id(
                base_metadata.source,
                chunk_text,
            )

            chunk_metadata = ChunkMetadata(
                topic=base_metadata.topic,
                difficulty=base_metadata.difficulty,
                type=base_metadata.type,
                source=base_metadata.source,
                related_topics=base_metadata.related_topics.copy(),
                is_bonus=base_metadata.is_bonus,
            )

            document_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    metadata=chunk_metadata,
                )
            )

        logger.info("Chunked {} into {} chunks", file_path.name, len(document_chunks))
        return document_chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk multiple files in a single call.

        Used by the UI multi-file upload handler to process all
        uploaded files before passing to VectorStoreManager.ingest().

        Parameters
        ----------
        file_paths : list[Path]
            List of file paths to process.
        metadata_overrides : dict, optional
            Applied to all files. Per-file metadata should be handled
            by calling chunk_file() individually.

        Returns
        -------
        list[DocumentChunk]
            Combined chunks from all files, preserving source attribution
            in each chunk's metadata.
        """
        # TODO: implement — iterate and collect, handle per-file errors
        all_chunks: list[DocumentChunk] = []

        for file_path in file_paths:
            try:
                chunks = self.chunk_file(
                    file_path=file_path,
                    metadata_overrides=metadata_overrides,
                )
                all_chunks.extend(chunks)
            except Exception as e:
                logger.exception("Failed to chunk file {}", file_path)
                raise RuntimeError(f"Failed to chunk {file_path.name}: {str(e)}") from e

        logger.info("Prepared {} total chunks from {} files", len(all_chunks), len(file_paths))
        return all_chunks

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a PDF file.

        Uses PyPDFLoader for text extraction followed by
        RecursiveCharacterTextSplitter for chunking.

        Interview talking point: PDFs from academic papers often contain
        noisy content (headers, footers, reference lists, equations as
        text). Post-processing to remove this noise improves retrieval
        quality significantly.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'page' keys before conversion
            to DocumentChunk objects.
        """
        # TODO: implement using langchain_community.document_loaders.PyPDFLoader
        # and langchain.text_splitter.RecursiveCharacterTextSplitter
        from langchain.text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        split_docs = splitter.split_documents(documents)

        return [
            {
                "text": doc.page_content,
                "page": doc.metadata.get("page"),
            }
            for doc in split_docs
            if doc.page_content and doc.page_content.strip()
        ]

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a Markdown file.

        Uses MarkdownHeaderTextSplitter first to respect document
        structure (headers create natural chunk boundaries), then
        RecursiveCharacterTextSplitter for oversized sections.

        Interview talking point: header-aware splitting preserves
        semantic coherence better than naive character splitting —
        a concept within one section stays within one chunk.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'header' keys.
        """
        # TODO: implement using langchain.text_splitter.MarkdownHeaderTextSplitter
        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()

        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "header": file_path.stem,
                    }
                )

            if end == text_len:
                break

            start = max(end - chunk_overlap, start + 1)

        return chunks

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """
        Infer chunk metadata from filename conventions and apply overrides.

        Filename convention (recommended to Corpus Architects):
          <topic>_<difficulty>.md or <topic>_<difficulty>.pdf
          e.g. lstm_intermediate.md, alexnet_advanced.pdf

        If the filename does not follow this convention, defaults are
        applied and the Corpus Architect must provide overrides manually.

        Parameters
        ----------
        file_path : Path
            Source file path used to infer topic and difficulty.
        overrides : dict, optional
            Explicit metadata values that take precedence over inference.

        Returns
        -------
        ChunkMetadata
            Populated metadata object.
        """
        # TODO: implement filename parsing + override merging
        # Bonus topics: SOM, BoltzmannMachine, GAN → set is_bonus=True
        stem_parts = file_path.stem.split("_")

        topic_map = {
            "ann": "ANN",
            "cnn": "CNN",
            "rnn": "RNN",
            "lstm": "LSTM",
            "seq2seq": "Seq2Seq",
            "autoencoder": "Autoencoder",
            "som": "SOM",
            "boltzmann": "BoltzmannMachine",
            "gan": "GAN",
        }

        inferred_topic = topic_map.get(stem_parts[0].lower(), "ANN")
        inferred_difficulty = (
            stem_parts[1].lower()
            if len(stem_parts) > 1 and stem_parts[1].lower() in {"beginner", "intermediate", "advanced"}
            else "intermediate"
        )

        related_topics_map = {
            "ANN": ["backpropagation", "activation_functions"],
            "CNN": ["convolution", "pooling", "feature_maps"],
            "RNN": ["sequence_modeling", "hidden_state", "vanishing_gradient"],
            "LSTM": ["RNN", "gates", "long_term_dependencies"],
            "Seq2Seq": ["encoder_decoder", "translation", "RNN"],
            "Autoencoder": ["representation_learning", "dimensionality_reduction"],
            "SOM": ["unsupervised_learning", "clustering"],
            "BoltzmannMachine": ["energy_based_models"],
            "GAN": ["generator", "discriminator"],
        }

        data = {
            "topic": inferred_topic,
            "difficulty": inferred_difficulty,
            "type": "concept_explanation",
            "source": file_path.name,
            "related_topics": related_topics_map.get(inferred_topic, []),
            "is_bonus": inferred_topic in {"SOM", "BoltzmannMachine", "GAN"},
        }

        if overrides:
            data.update(overrides)

        return ChunkMetadata(
            topic=data["topic"],
            difficulty=data["difficulty"],
            type=data["type"],
            source=data["source"],
            related_topics=data.get("related_topics", []),
            is_bonus=data.get("is_bonus", False),
    )
