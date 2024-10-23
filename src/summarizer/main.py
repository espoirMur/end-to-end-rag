# Main summarize pipeline

from src.summarizer.cluster_modeler import HierachicalClusterModeler
from src.summarizer.data_puller import DataPuller
from datetime import datetime

if __name__ == "__main__":
    today = datetime.now().strftime("%Y-%m-%d")
    data_puller = DataPuller(environment="local")
    data = data_puller.read_data(today)
    embedding_model_id = "dunzhang/stella_en_400M_v5"
    clustering = HierachicalClusterModeler(
        documents=data, embedding_model_id=embedding_model_id)
    clustering.run()
    data_puller.save_data(clustering.documents)
