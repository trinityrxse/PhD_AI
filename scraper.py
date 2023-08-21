# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import requests
import os
from bs4 import BeautifulSoup
import operator

def get_arXiv_id(researcher="researcher"):
    # Set up search parameters
    search_phrase = f'{researcher}'  # Researcher name must have + instead of space
    search_url = 'https://arxiv.org/search/?query=' + search_phrase.replace(' ',
                                                                            '+') + '&searchtype=author&abstracts=show&order=-announced_date_first&size=50'
    # Make request to search URL
    response = requests.get(search_url)

    # Parse HTML of search page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all links to papers on search page
    paper_links = soup.find('p', class_='list-title is-inline-block').text.strip()
    length = len(paper_links)
    N = 12
    # create a new string of last N characters
    id = operator.getitem(paper_links, slice(6, length - N))

    return id

get_arXiv_id('K S Stelle')
