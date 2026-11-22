# Hasil evaluasi (kueri TEST)

| Konfigurasi | nDCG@10 | MAP@100 | Recall@100 | P@10 | Parameter |
|---|---|---|---|---|---|
| BM25 | 0.4009 | 0.3400 | 0.8571 | 0.1396 | - |
| BM25+stem | 0.4201 | 0.3587 | 0.8794 | 0.1410 | - |
| Dense | 0.5075 | 0.4328 | 0.8732 | 0.1635 | - |
| Hybrid-RRF | 0.5514 | 0.4764 | 0.9332 | 0.1873 | rrf_k=10 |
| Hybrid-RRF+stem | 0.5605 | 0.4865 | 0.9417 | 0.1865 | rrf_k=10 |
| Hybrid-Linear | 0.5613 | 0.4883 | 0.9291 | 0.1875 | alpha=0.5 |
| Hybrid-Linear+stem | 0.5731 | 0.5008 | 0.9380 | 0.1860 | alpha=0.5 |

## Uji signifikansi (nDCG@10)

| A | B | delta(A-B) | CI95_bawah | CI95_atas | p | p_holm | signifikan(0.05) |
|---|---|---|---|---|---|---|---|
| Hybrid-RRF | BM25 | 0.1505 | 0.1289 | 0.1726 | 0.0001 | 0.0013 | True |
| Hybrid-RRF | Dense | 0.0439 | 0.0247 | 0.0640 | 0.0001 | 0.0013 | True |
| Hybrid-Linear | BM25 | 0.1604 | 0.1380 | 0.1833 | 0.0001 | 0.0013 | True |
| Hybrid-Linear | Dense | 0.0538 | 0.0350 | 0.0733 | 0.0001 | 0.0013 | True |
| Hybrid-RRF+stem | BM25+stem | 0.1404 | 0.1188 | 0.1621 | 0.0001 | 0.0013 | True |
| Hybrid-RRF+stem | Dense | 0.0530 | 0.0343 | 0.0724 | 0.0001 | 0.0013 | True |
| Hybrid-Linear+stem | BM25+stem | 0.1529 | 0.1308 | 0.1751 | 0.0001 | 0.0013 | True |
| Hybrid-Linear+stem | Dense | 0.0656 | 0.0459 | 0.0856 | 0.0001 | 0.0013 | True |
| Hybrid-Linear | Hybrid-RRF | 0.0099 | 0.0016 | 0.0183 | 0.0171 | 0.0684 | False |
| Hybrid-Linear+stem | Hybrid-RRF+stem | 0.0126 | 0.0040 | 0.0213 | 0.0041 | 0.0205 | True |
| BM25+stem | BM25 | 0.0192 | 0.0026 | 0.0364 | 0.0283 | 0.0849 | False |
| Hybrid-RRF+stem | Hybrid-RRF | 0.0091 | -0.0019 | 0.0207 | 0.1141 | 0.1141 | False |
| Hybrid-Linear+stem | Hybrid-Linear | 0.0117 | 0.0007 | 0.0232 | 0.0441 | 0.0882 | False |

## Latensi per kueri

| Konfigurasi | median_ms | mean_ms | p95_ms |
|---|---|---|---|
| BM25 | 6.7 | 6.9 | 9.7 |
| BM25+stem | 6.9 | 6.7 | 8.2 |
| Dense | 55.5 | 56.6 | 64.1 |
| Hybrid-RRF | 62.6 | 62.6 | 68.1 |
| Hybrid-RRF+stem | 61.9 | 62.2 | 68.4 |
| Hybrid-Linear | 61.9 | 62.1 | 67.5 |
| Hybrid-Linear+stem | 61.6 | 62.1 | 67.7 |