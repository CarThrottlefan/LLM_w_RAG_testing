from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.retriever.dense.Contriever import Contriever
from dexter.utils.metrics.SimilarityMatch import DotScore
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams

config_instance = DenseHyperParams()
loader = DprDataLoader("wikimultihopqa", config_path="D:\\NLP_proj_LLM_w-RAG\\config.ini", config=config_instance)
queries, qrels, corpus = loader.qrels()
retriever = DprSentSearch(config_instance, )
similarity_measure = DotScore()