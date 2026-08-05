import numpy as np

from vectordb.document_store import DocumentStore
from vectordb.embedder import AilaEmbedder
from vectordb.faiss_index import FaissIndex
from vectordb.semantic_index import SemanticIndex


def test_embedder_output_shape_and_normalization(tiny_model, tokenizer):
    embedder = AilaEmbedder(tiny_model, tokenizer)
    vecs = embedder.embed(["hello world", "a longer sentence with more tokens in it"])
    assert vecs.shape == (2, tiny_model.cfg.d_model)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_embedder_single_string_input(tiny_model, tokenizer):
    embedder = AilaEmbedder(tiny_model, tokenizer)
    vec = embedder.embed("hello")
    assert vec.shape == (1, tiny_model.cfg.d_model)


def test_faiss_index_add_search_remove():
    idx = FaissIndex(dim=4)
    vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    ids = np.array([10, 20, 30])
    idx.add(vecs, ids)
    assert idx.ntotal == 3

    scores, result_ids = idx.search(np.array([[1, 0, 0, 0]], dtype=np.float32), k=2)
    assert result_ids[0][0] == 10

    idx.remove(np.array([10]))
    assert idx.ntotal == 2


def test_faiss_index_save_load(tmp_path):
    idx = FaissIndex(dim=3)
    idx.add(np.eye(3, dtype=np.float32), np.array([1, 2, 3]))
    path = str(tmp_path / "index.faiss")
    idx.save(path)

    loaded = FaissIndex.load(path, dim=3)
    assert loaded.ntotal == 3


def test_document_store_crud(tmp_path):
    store = DocumentStore(str(tmp_path / "docs.db"))
    doc_id = store.add("hello world", metadata={"source": "test"})
    doc = store.get(doc_id)
    assert doc["text"] == "hello world"
    assert doc["metadata"]["source"] == "test"

    store.delete(doc_id)
    assert store.get(doc_id) is None


def test_semantic_index_search_ranks_relevant_doc_first(tiny_model, tokenizer, tmp_path):
    embedder = AilaEmbedder(tiny_model, tokenizer)
    idx = SemanticIndex(
        embedder, db_path=str(tmp_path / "docs.db"), faiss_path=str(tmp_path / "idx.faiss")
    )
    ids = idx.add_documents(
        [
            "Aila Nano is a small language model.",
            "The weather today is sunny and warm.",
        ]
    )
    assert len(idx) == 2

    results = idx.search("Tell me about language models", k=2)
    assert len(results) == 2
    assert results[0]["id"] in ids

    idx.delete(ids[0])
    assert len(idx) == 1
