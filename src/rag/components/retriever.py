from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Any, Tuple
from rag.shared.database import execute_query, generate_database_connection
from spacy.language import Language
from textacy import extract
import spacy
from unicodedata import normalize as unicode_normalize


class HybridRetriever:

    """This class will perform hybrid retrieval, a combination of semantic search and keyword search"""

    def __init__(self, model_id: str, spacy_model: str):

        sentence_transformer_model = SentenceTransformer(model_id)
        cross_encoder = CrossEncoder(model_id)
        spacy_model: Language = spacy.load(spacy_model)
        self.database_connection = generate_database_connection()
        self.sentence_transformer_model = sentence_transformer_model
        self.cross_encoder = cross_encoder
        self.spacy_model = spacy_model

    def semantic_search(self, query: str) -> List[Any]:
        """this function perporm semmantic searc"""
        embedding = self.sentence_transformer_model.encode(query)
        semantic_search_query = 'SELECT id, chunk FROM article_embeddings ORDER BY chunk_vector <=> %(embedding)s LIMIT 5'
        results = execute_query(self.database_connection, semantic_search_query, {
                                'embedding': str(embedding.tolist())})
        return results

    def keyword_search(self, query: str) -> List[Any]:
        """This function will perform keyword search"""
        keyword_search_query_string = """SELECT article_id, chunk 
                                    FROM article_embeddings, websearch_to_tsquery(%(language)s, %(query)s) query
                                      WHERE to_tsvector(%(language)s, chunk) @@ query 
                                    ORDER BY ts_rank_cd(to_tsvector(%(language)s, chunk), query) DESC LIMIT %(limit)s;"""
        results = execute_query(self.database_connection, keyword_search_query_string, {
                                'language': 'unaccent_french', 'query': query, 'limit': 5})
        return results

    def perform_keyword_extraction(self, text: str) -> str:
        """This function will perform keyword extraction the text supplied.
        It used spacy and texacy and will perform keyword exraction and will return those top keywords ready to be used in websearch_text 
        function.
        The keywords will be combined with 'or' operator.
        """
        spacy_doc = self.spacy_model(text)
        term_keys = extract.keyterms.textrank(
            spacy_doc, normalize="lemma", topn=3)
        return " or ".join([f'"{term[0]}"' for term in term_keys])

    def rerank(self, query: str, results: List[Tuple[int, str]]) -> List[Any]:
        """ this function rerank the results based on the their similarity with the question"""
        results = [result[1] for result in results]
        results = set(results)
        # re-rank
        scores = self.cross_encoder.predict(
            [(query, item) for item in results])
        return [v for _, v in sorted(zip(scores, results), reverse=True)]

    def run(self, query: str) -> List[Any]:
        """This function will run the hybrid retriever and will return the results"""
        semantic_results = self.semantic_search(query)
        keywords = self.perform_keyword_extraction(query)
        keyword_results = self.keyword_search(keywords)
        results = semantic_results + keyword_results
        re_ranked_results = self.rerank(query, results)
        re_ranked_results = [unicode_normalize(
            'NFC', result) for result in re_ranked_results]
        return re_ranked_results
