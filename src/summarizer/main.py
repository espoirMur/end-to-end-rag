# Main summarize pipeline

from src.summarizer.cluster_modeler import HierachicalClusterModeler
from src.summarizer.data_puller import DataPuller
from datetime import datetime, timedelta
import argparse


parser = argparse.ArgumentParser(
    prog='new summarizer', description='a news summarizer')
if __name__ == "__main__":
    # read arg named environment form the command line
    parser.add_argument('-e', '--environment', default="dev")
    parser.add_argument('-d', '--days_ago', type=int, default=1)
    parser.add_argument("-st", "--storage_mode", default="local")
    args = parser.parse_args()
    environment = args.environment
    days_ago = args.days_ago
    date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

    data_puller = DataPuller(environment=environment, date=date)
    data = data_puller.read_data()
    embedding_model_id = "dunzhang/stella_en_400M_v5"
    clustering = HierachicalClusterModeler(
        documents=data, embedding_model_id=embedding_model_id)
    clustering.run()
    data_puller.save_data(clustering.documents, storage_mode=args.storage_mode)
