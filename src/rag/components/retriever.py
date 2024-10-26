from sentence_transformers import SentenceTransformer
from src.shared.custom_cross_encoder import CustomCrossEncoder
from typing import List, Any, Tuple
from src.shared.database import execute_query, generate_database_connection
from spacy.language import Language
from textacy import extract
import spacy
from unicodedata import normalize as unicode_normalize


class HybridRetriever:

    """This class will perform hybrid retrieval, a combination of semantic search and keyword search"""

    def __init__(self, cross_encoder_kwargs: dict, spacy_model: str, language: str, sentence_transformer_kwargs: dict):

        sentence_transformer_model = SentenceTransformer(
            **sentence_transformer_kwargs)

        cross_encoder = CustomCrossEncoder(**cross_encoder_kwargs)

        spacy_model: Language = spacy.load(spacy_model)
        self.database_connection = generate_database_connection()
        self.sentence_transformer_model = sentence_transformer_model
        self.cross_encoder = cross_encoder
        self.spacy_model = spacy_model
        self.language = language

    def semantic_search(self, query: str, limit: int = 5) -> List[Any]:
        """
        Query the database for semantic search:
        transform the query string to the embeddings
        and then retrieve the embedding that are similar to the query string.
        """
        embedding = self.sentence_transformer_model.encode(query)
        semantic_search_query = 'SELECT id, content FROM haystack_documents ORDER BY embedding <=> %(embedding)s LIMIT %(limit)s'
        results = execute_query(self.database_connection, semantic_search_query, {
                                'embedding': str(embedding.tolist()), 'limit': limit})
        return results

    def keyword_search(self, keywords: str, limit: int = 5) -> List[Any]:
        """This function will perform keyword search"""
        keyword_search_query_string = """SELECT id, content 
                                    FROM haystack_documents, websearch_to_tsquery(%(language)s, %(keywords)s) query
                                      WHERE to_tsvector(%(language)s, content) @@ query 
                                    ORDER BY ts_rank_cd(to_tsvector(%(language)s, content), query) DESC LIMIT %(limit)s;"""
        results = execute_query(self.database_connection, keyword_search_query_string, {
                                'language': self.language, 'keywords': keywords, 'limit': limit})
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
