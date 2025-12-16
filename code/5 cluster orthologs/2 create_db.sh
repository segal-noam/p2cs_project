# create database for HKs by amino acids
mmseqs createdb /zdata/user-data/noam/data/p2cs/merged_p2cs_data/fasta/hk_filtered_groups.faa /zdata/user-data/noam/data/p2cs/clusters/mmseqs_db/hkAA_DB
# create database for RRs by amino acids
mmseqs createdb /zdata/user-data/noam/data/p2cs/merged_p2cs_data/fasta/rr_filtered_groups.faa /zdata/user-data/noam/data/p2cs/clusters/mmseqs_db/rrAA_DB

# create database for HKs by nucleotides
mmseqs createdb /zdata/user-data/noam/data/p2cs/merged_p2cs_data/fasta/hk_filtered_groups.fna /zdata/user-data/noam/data/p2cs/clusters/mmseqs_db/hkNT_DB
# create database for RRs by nucleotides
mmseqs createdb /zdata/user-data/noam/data/p2cs/merged_p2cs_data/fasta/rr_filtered_groups.fna /zdata/user-data/noam/data/p2cs/clusters/mmseqs_db/rrNT_DB