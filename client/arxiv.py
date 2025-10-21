

class ArXivClient():
    """ 
    TODO:
    
    - ref: https://info.arxiv.org/help/api/user-manual.html#52-details-of-atom-results-returned
        - (details of atom results returned)
    - this would be an arXiv client for:
        - pdf downloads for a given arxiv_id
        - fetch associated metadata like:
            - <title> element
            - <id> canonical URL identifier of paper
            - <summary> element which is the abstract
            - <author> for authors
            - <published> date published
            - <arxiv:affiliation> author affiliation 
            - <arxiv:doi> url for the resolved DOI to an external resource if present
            - <category> subject classification (e.g., Human-Computer Interaction)
            - <arxiv:primary_category> subject classification (primary) specific from arXiv (https://arxiv.org/category_taxonomy)
    - inherits from a `BaseClient`
    
    
    """
    pass
