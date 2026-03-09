import os
from pathlib import Path
from src.utils.logger import logger

class StorageService:
    def __init__(self, base_path: str):
        # Convert string path from config to a Path object
        self.base_path = Path(base_path)
        # Create the directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_paper_metadata(self, paper_id: str, content: str):
        """
        Saves the paper metadata to a file. 
        This name must match exactly what pipeline.py calls.
        """
        file_path = self.base_path / f"{paper_id}.txt"
        
        # Production check: don't overwrite if it exists
        if file_path.exists():
            logger.info(f"File {paper_id}.txt already exists. Skipping.")
            return
            
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Successfully saved: {paper_id}.txt")
        except Exception as e:
            logger.error(f"Could not save file {paper_id}: {e}")
    def save_paper_metadata(self, paper_id: str, content: str) -> bool:
        file_path = self.base_path / f"{paper_id}.txt"
        
        if file_path.exists():
            return False # Signal that we skipped it
            
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True # Signal a successful new save
        except Exception as e:
            logger.error(f"Error saving {paper_id}: {e}")
            return False