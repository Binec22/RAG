import argparse
import os
import shutil
from typing import List
from pathlib import Path
import logging

from tqdm import tqdm
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.RagClasses.CustomEmbeddings import CustomEmbeddings
from app.RagClasses import AppConfig


class Database:
    def __init__(self, embedding = None, config=None):
        """
        Initialize the database with optional configuration

        Args:
            config_name (str, optional): Name of configuration to load
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # Private
        self._data_files_path = None
        self._embedded_database_path = None
        self._embedding_model_name = None

        # Config publique mais attributs privés pour plus de clarté à l'intérieur de la classe
        self.config = config
        self.load_config()

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = CustomEmbeddings(model_name=self._embedding_model_name).model

        self.database = Chroma(
            persist_directory=self._find_chroma_path(),
            embedding_function=self.embedding
        )


    def load_config(self):
        """
        Load configuration parameters from config
        """
        self._data_files_path = self.config["data_files_path"]
        self._embedded_database_path = self.config["embedded_database_path"]
        self._embedding_model_name = self.config["embedding_model"]

    def add_to_chroma(self, chunks: list[Document]):
        """
        Add document chunks to Chroma database

        Args:
            chunks (list): Document chunks to add
        """
        # Calculate and add chunk IDs
        chunks_with_ids = self._calculate_chunk_ids(chunks)

        # Check for existing documents
        existing_items = self.database.get(include=[])
        existing_ids = set(existing_items["ids"])

        print(f"Number of existing documents in Database: {len(existing_ids)}")

        new_chunks = [
            chunk for chunk in chunks_with_ids
            if chunk.metadata["id"] not in existing_ids
        ]

        batch_size = 1000

        if new_chunks:
            with tqdm(total=len(new_chunks), desc="Adding documents") as pbar:
                for i in range(0, len(new_chunks), batch_size):
                    batch = new_chunks[i:i + batch_size]
                    new_chunk_ids = [chunk.metadata["id"] for chunk in batch]
                    self.database.add_documents(batch, ids=new_chunk_ids)
                    pbar.update(len(batch))
        else:
            print("No new documents to add")

    def detect_file_types(self) -> List[str]:
        """
        Detect unique file types in the data directory.

        Returns:
            List[str]: List of unique file extensions found.
        """
        file_types = set()
        print(self._data_files_path)
        for root, _, files in os.walk(self._data_files_path):
            for file in files:
                print(file)
                file_types.add(os.path.splitext(file)[1].lower())
        return list(file_types)

    def load_txt_docx_documents(self) -> List[Document]:
        """
        Load TXT and DOCX documents using SimpleDirectoryReader.

        Returns:
            List[Document]: Loaded TXT and DOCX documents.
        """
        from llama_index.core import SimpleDirectoryReader

        documents = []
        llama_document_loader = SimpleDirectoryReader(
            input_dir=self._data_files_path,
            required_exts=[".txt", ".docx"]
        )
        for doc in tqdm(llama_document_loader.load_data(), desc="Loading .txt and .docx documents"):
            doc.metadata.pop('file_path', None)
            documents.append(doc)
        return documents

    def load_pdf_documents(self) -> List[Document]:
        """
        Load PDF documents using ProgressPyPDFDirectoryLoader.

        Returns:
            List[Document]: Loaded PDF documents.
        """
        pdf_loader = ProgressPyPDFDirectoryLoader(self._data_files_path)
        documents = []
        for doc in tqdm(pdf_loader.load(), desc="Loading .pdf documents"):
            documents.append(doc)
        return documents

    def load_json_documents(self) -> List[Document]:
        """
        Load .json documents using JSONLoader.

        Returns:
            List[Document]: Loaded PDF documents.
        """
        from langchain_community.document_loaders import JSONLoader

        data_files_path = Path(self._data_files_path)

        def metadata_func(record: dict, metadata: dict) -> dict:
            metadata["url"] = record.get("url")
            metadata["key_words"] = record.get("key_words")
            return metadata

        json_files = list(data_files_path.glob("*.json"))

        documents = []

        for json_file in tqdm(json_files, desc="Loading JSON documents"):
            json_loader = JSONLoader(
                file_path=str(json_file),
                jq_schema=".[] | .data",
                content_key="text_content",
                metadata_func=metadata_func,
                text_content=False
            )

            # Ajouter les documents chargés à la liste
            documents.extend(json_loader.load())

        return documents

    def load_documents(self) -> List[Document]:
        """
        Load documents from various file types and combine them.

        Returns:
            List[Document]: Combined list of all loaded documents.
        """
        file_types = self.detect_file_types()
        all_documents = []

        if ".txt" in file_types or ".docx" in file_types:
            all_documents.extend(self.load_txt_docx_documents())

        if ".pdf" in file_types:
            all_documents.extend(self.load_pdf_documents())

        if ".json" in file_types:
            all_documents.extend(self.load_json_documents())

        print(f"Loaded {len(all_documents)} documents from {len(file_types)} file types.")
        return all_documents

    @staticmethod
    def _convert_llamaindexdoc_to_langchaindoc(documents: list):
        """
        Convert llamaindex documents to langchain documents

        Args:
            documents (list): List of llamaindex documents

        Returns:
            list: Converted langchain documents
        """
        return [
            Document(page_content=doc.text, metadata=doc.metadata)
            for doc in documents
        ]

    @staticmethod
    def split_documents(documents: list[Document]):
        """
        Split documents into smaller chunks

        Args:
            documents (list): Documents to split

        Returns:
            list: Split document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    @staticmethod
    def _calculate_chunk_ids(chunks):
        """
        Generate unique IDs for document chunks

        Args:
            chunks (list): Document chunks

        Returns:
            list: Chunks with added ID metadata
        """
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("file_name")
            page = str(chunk.metadata.get("page"))
            current_page_id = f"{source}:p{page}"

            current_chunk_index = (
                current_chunk_index + 1
                if current_page_id == last_page_id
                else 0
            )

            chunk.metadata["id"] = f"{current_page_id}:c{current_chunk_index}"
            last_page_id = current_page_id

        return chunks

    def _find_chroma_path(self):
        """
        Find or create Chroma database path for specific embedding model

        Returns:
            str: Path to Chroma database
        """
        model_path = os.path.join(
            self._embedded_database_path,
            f"chroma_{self._embedding_model_name}"
        )

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        return model_path

    def clear_database(self, chroma_subfolder_name=None):
        """
        Clear entire database or specific subfolder

        Args:
            chroma_subfolder_name (str, optional): Specific subfolder to clear
        """
        if chroma_subfolder_name:
            full_path = os.path.join(
                self._embedded_database_path,
                chroma_subfolder_name
            )
            if os.path.exists(full_path):
                shutil.rmtree(full_path)
                print(f"Database in {full_path} deleted.")
            else:
                raise FileNotFoundError(f"Folder {full_path} doesn't exist")
        else:
            print("\nExisting databases:\n")
            subfolders = [
                f for f in os.listdir(self._embedded_database_path)
                if os.path.isdir(os.path.join(self._embedded_database_path, f))
            ]

            if not subfolders:
                print(f"No subfolders found in {self._embedded_database_path}\n")
                return

            for subfolder in subfolders:
                print(f"- {subfolder}")

            confirmation = input("Delete all databases? (yes/no): ")
            if confirmation.lower() == 'yes':
                shutil.rmtree(self._embedded_database_path)
                print("All databases cleared")
            else:
                print("Deletion cancelled.")

    def populate_database(self):
        """
        Full database population workflow
        """
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        self.add_to_chroma(chunks)

    @classmethod
    def run_from_cli(cls):
        """
        CLI entry point for database management
        """
        parser = argparse.ArgumentParser(description="Database management script")
        parser.add_argument("--config", type=str, help="Configuration to load")
        parser.add_argument("--reset", action="store_true", help="Reset database")
        parser.add_argument("--clear", action="store_true", help="Clear database")

        args = parser.parse_args()
        config = AppConfig.from_json(config_name=args.config).as_dict()
        print(config)
        db = cls(config=config)

        if args.clear:
            print("Clearing Database...")
            subfolder_name = (
                f"chroma_{db._embedding_model_name}"
                if args.config else None
            )
            db.clear_database(subfolder_name)
            return

        if args.config:
            if args.reset:
                print("Resetting Database...")
                subfolder_name = f"chroma_{db._embedding_model_name}"
                db.clear_database(subfolder_name)

            db.populate_database()


class ProgressPyPDFDirectoryLoader(PyPDFDirectoryLoader):
    """Enhanced PDF directory loader with progress tracking"""
    def load(self) -> List[Document]:
        p = Path(self.path)
        docs = []
        items = list(p.rglob(self.glob)) if self.recursive else list(p.glob(self.glob))

        with tqdm(total=len(items), desc="Loading PDFs") as pbar:
            for i in items:
                if i.is_file():
                    if self._is_visible(i.relative_to(p)) or self.load_hidden:
                        try:
                            loader = PDFPlumberLoader(str(i), extract_images=self.extract_images)
                            sub_docs = loader.load()
                            for doc in sub_docs:
                                if 'source' in doc.metadata:
                                    doc.metadata['source'] = i.name
                                    doc.metadata['file_name'] = doc.metadata.pop('source')
                                doc.metadata.pop('file_path', None)
                                doc.metadata = {
                                    key: value for key, value in doc.metadata.items()
                                    if value
                                }
                            docs.extend(sub_docs)
                        except Exception as e:
                            if self.silent_errors:
                                logging.warning(e)
                            else:
                                raise e
                pbar.update(1)
        return docs


if __name__ == "__main__":
    Database.run_from_cli()