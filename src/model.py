from typing import Callable

"""
Base class for categorizing companies.
This class defines the interface for categorizing companies based on their names and descriptions.
"""

class Model:
    def __init__(self):
        pass

    def categorize(self, company_name: str, description: str = None) -> str:
        """
        Categorize a company based on its name and description.
        
        Args:
            company_name (str): The name of the company.
            description (str, optional): The description of the company. Defaults to None.
        
        Returns:
            str: The category of the company.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def categorize_get_description(self, company_name: str, get_description: Callable[[str], str]) -> str:
        """
        Categorize a company based on its name and a description obtained from a function.
        
        Args:
            company_name (str): The name of the company.
            get_description (Callable[[str], str]): A function that takes a company name and returns its description.
        
        Returns:
            str: The category of the company.
        """
        description = get_description(company_name)
        return self.categorize(company_name, description)