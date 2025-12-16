# cluster HKs by amino acids with 90% identity
mmseqs cluster /zdata/user-data/noam/data/p2cs/clusters/mini_sample_mmseqs_db/hkAA_DB \
    /zdata/user-data/noam/data/p2cs/clusters/mini_sample_mmseqs_db/hkAA_DB_clu \
    /zdata/user-data/noam/data/p2cs/clusters/mini_sample_mmseqs_db/hkAA_DB_tmp \
    --min-seq-id 0.9 \
    -c 0.9 \
    --cov-mode 2 \
    --alignment-mode 3 \
    --threads 8 \
    --cluster-mode 0

mmseqs createtsv /zdata/user-data/noam/data/p2cs/clusters/mini_sample_mmseqs_db/hkAA_DB \
    /zdata/user-data/noam/data/p2cs/clusters/mini_sample_mmseqs_db/hkAA_DB \
    /zdata/user-data/noam/data/p2cs/clusters/mini_sample_mmseqs_db/hkAA_DB_clu \
    /zdata/user-data/noam/data/p2cs/clusters/mini_sample_hkAA_DB_clu.tsv

# cluster RRs by amino acids with 90% identity
mmseqs cluster /zdata/user-data/noam/data/p2cs/clusters/mini_sample_mmseqs_db/rrAA_DB \
    /zdata/user-data/noam/data/p2cs/clusters/mini_sample_mmseqs_db/rrAA_DB_clu \
    /zdata/user-data/noam/data/p2cs/clusters/mini_sample_mmseqs_db/rrAA_DB_tmp \
    --min-seq-id 0.9 \
    -c 0.9 \
    --cov-mode 2 \
    --alignment-mode 3 \
    --threads 8 \
    --cluster-mode 0

mmseqs createtsv /zdata/user-data/noam/data/p2cs/clusters/mini_sample_mmseqs_db/rrAA_DB \
    /zdata/user-data/noam/data/p2cs/clusters/mini_sample_mmseqs_db/rrAA_DB \
    /zdata/user-data/noam/data/p2cs/clusters/mini_sample_mmseqs_db/rrAA_DB_clu \
    /zdata/user-data/noam/data/p2cs/clusters/mini_sample_rrAA_DB_clu.tsv