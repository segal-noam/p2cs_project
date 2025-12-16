import requests
from bs4 import BeautifulSoup
import time
import os
from tqdm import tqdm
import pandas as pd
import glob

# --------------------------------------------

# The base session page with the dropdown
export_page_url = "http://www.p2cs.org/index.php?section=export"

# Start a session to preserve cookies and PHPSESSID
session = requests.Session()

# Fetch the export page
response = session.get(export_page_url)
soup = BeautifulSoup(response.text, "html.parser")

# Extract all organism options from the <select name="exportDB"> dropdown
options = soup.find("select", {"name": "exportDB"}).find_all("option")
organism_db_ids = []
for opt in options:
    val = opt.get('value')
    if val:
        # Get only the first NavigableString before the <br> or other tags
        label = opt.contents[0].strip() if opt.contents else ''
        organism_db_ids.append((val, label))

print(f"Found {len(organism_db_ids)} organisms")

# --------------------------------------------

# Download directory
os.makedirs("p2cs_exports", exist_ok=True)

for db_id, org_name in tqdm(organism_db_ids):
    # export_url = f"http://www.p2cs.org/exportTCSInTextFile.php?exportDB={db_id}"
    aa_export_url = f"http://www.p2cs.org/exportTCSInTextFile.php?exportDB={db_id}&fasta=p"
    nt_export_url = f"http://www.p2cs.org/exportTCSInTextFile.php?exportDB={db_id}&fasta=n"
    try:
        # export_resp = session.get(export_url)
        aa_export_resp = session.get(aa_export_url)
        nt_export_resp = session.get(nt_export_url)
        # Check and save amino acid (protein) export
        if aa_export_resp.ok and len(aa_export_resp.content) > 100:
            if '/' in org_name:
                aa_filename = f"p2cs_exports/{db_id}_aa.fasta"
            else:
                aa_filename = f"p2cs_exports/{db_id}_{org_name.replace(' ', '_')}_aa.fasta"
            with open(aa_filename, "wb") as f:
                f.write(aa_export_resp.content)
        else:
            print(f"⚠️ Empty or failed AA response for {org_name}")

        # Check and save nucleotide export
        if nt_export_resp.ok and len(nt_export_resp.content) > 100:
            if '/' in org_name:
                nt_filename = f"p2cs_exports/{db_id}_nt.fasta"
            else:
                nt_filename = f"p2cs_exports/{db_id}_{org_name.replace(' ', '_')}_nt.fasta"
            with open(nt_filename, "wb") as f:
                f.write(nt_export_resp.content)
        else:
            print(f"⚠️ Empty or failed NT response for {org_name}")
    except Exception as e:
        print(f"❌ Error downloading {org_name}: {e}")
        time.sleep(1)  # Be polite to the server

# --------------------------------------------

# # Folder with downloaded .txt files
# folder = 'p2cs_exports'
# file_paths = glob.glob(os.path.join(folder, '*.txt'))

# dfs = []
# common_cols = None

# for file_path in file_paths:
#     try:
#         df = pd.read_csv(file_path, sep='\t', engine='python', on_bad_lines='skip')

#         if df.empty or len(df.columns) == 0:
#             print(f"⚠️ Skipping empty or invalid file: {file_path}")
#             continue

#         # Extract organism name from filename (without .txt)
#         organism_name = os.path.splitext(os.path.basename(file_path))[0]
#         df["organism"] = organism_name

#         # Track shared columns
#         if common_cols is None:
#             common_cols = set(df.columns)
#         else:
#             common_cols &= set(df.columns)

#         dfs.append(df)

#     except Exception as e:
#         print(f"❌ Failed to load {file_path}: {e}")

# # Keep only common columns
# if common_cols:
#     dfs = [df[list(common_cols) + ["organism"]] for df in dfs]
#     combined_df = pd.concat(dfs, ignore_index=True)
#     print(f"✅ Combined {len(dfs)} files into one DataFrame with shape {combined_df.shape}")
# else:
#     print("❌ No common columns found across files.")

# Optional: Save to CSV
# combined_df.to_csv("combined_p2cs_data.tsv", sep='\t', index=False)

# ---------------------------------------------------------------------