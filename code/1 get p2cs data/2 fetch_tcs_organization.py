# -----------------------------------------------------------
# Imports

# progress bar
from tqdm import tqdm

# HTML parsing
from bs4 import BeautifulSoup

# file handling
import os
import glob

# data manipulation
import pandas as pd

# request handling
import requests

# regex
import re

# -----------------------------------------------------------

# Load the data
df_val_filtered = pd.read_pickle("/zdata/user-data/noam/data/p2cs/merged_p2cs_data/_p2cs_filtered_data.pkl")

num_rows = len(df_val_filtered)
# num_sections = 4
# section_size = num_rows // num_sections
# section_idx = 2

# Only process rows after the first 12000
start_idx = 12000
process_df = df_val_filtered.iloc[start_idx : ]

# -----------------------------------------------------------

def fetch_tcs_organization(db_id, gene, session=None):
    url = f"http://www.p2cs.org/getSequence.php?base={db_id}&gene={gene}"
    sess = session or requests
    response = sess.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    tcs_tag = soup.find('b', string="TCS organization")
    if tcs_tag:
        br = tcs_tag.find_next('br')
        if br:
            next_line = br.next_sibling
            if next_line:
                return next_line.strip() if isinstance(next_line, str) else next_line.get_text(strip=True)
    return None

# Use a session for all requests
with requests.Session() as sess:
    tqdm.pandas(desc="Fetching TCS organization")
    tcs_organization_list = []
    for idx, (row_idx, row) in enumerate(tqdm(process_df[['db_id', 'Gene']].iterrows(), total=len(process_df), desc="Fetching TCS organization")):
        tcs_org = fetch_tcs_organization(row['db_id'], row['Gene'], session=sess)
        tcs_organization_list.append(tcs_org)
        # Backup every 500 iterations
        if idx % 500 == 0:
            df_val_filtered.loc[process_df.index[:idx+1], 'tcs_organization'] = tcs_organization_list
            df_val_filtered.to_pickle(f"p2cs_filtered_data_{2}.pkl")
    # Assign the final results
    df_val_filtered.loc[process_df.index, 'tcs_organization'] = tcs_organization_list

# -----------------------------------------------------------
# Save the updated DataFrame
df_val_filtered.to_pickle(f"p2cs_filtered_data_{2}.pkl")
