# Hasil evaluasi (kueri TEST)

| Konfigurasi | nDCG@10 | MAP@100 | Recall@100 | P@10 | Parameter |
|---|---|---|---|---|---|
| BM25 | 0.4009 | 0.3400 | 0.8571 | 0.1396 | - |
| BM25+stem | 0.4201 | 0.3587 | 0.8794 | 0.1410 | - |
| Dense | 0.4942 | 0.4247 | 0.9115 | 0.1613 | - |
| Hybrid-RRF | 0.5206 | 0.4490 | 0.9367 | 0.1783 | rrf_k=5 |
| Hybrid-RRF+stem | 0.5367 | 0.4649 | 0.9420 | 0.1796 | rrf_k=10 |
| Hybrid-Linear | 0.5487 | 0.4726 | 0.9332 | 0.1862 | alpha=0.6 |
| Hybrid-Linear+stem | 0.5525 | 0.4763 | 0.9410 | 0.1838 | alpha=0.6 |

## Uji signifikansi (nDCG@10)

| A | B | delta(A-B) | CI95_bawah | CI95_atas | p | p_holm | signifikan(0.05) |
|---|---|---|---|---|---|---|---|
| Hybrid-RRF | BM25 | 0.1197 | 0.1001 | 0.1394 | 0.0001 | 0.0013 | True |
| Hybrid-RRF | Dense | 0.0264 | 0.0091 | 0.0435 | 0.0021 | 0.0105 | True |
| Hybrid-Linear | BM25 | 0.1477 | 0.1234 | 0.1723 | 0.0001 | 0.0013 | True |
| Hybrid-Linear | Dense | 0.0544 | 0.0403 | 0.0686 | 0.0001 | 0.0013 | True |
| Hybrid-RRF+stem | BM25+stem | 0.1166 | 0.0979 | 0.1347 | 0.0001 | 0.0013 | True |
| Hybrid-RRF+stem | Dense | 0.0425 | 0.0241 | 0.0615 | 0.0002 | 0.0013 | True |
| Hybrid-Linear+stem | BM25+stem | 0.1324 | 0.1102 | 0.1548 | 0.0001 | 0.0013 | True |
| Hybrid-Linear+stem | Dense | 0.0583 | 0.0441 | 0.0724 | 0.0001 | 0.0013 | True |
| Hybrid-Linear | Hybrid-RRF | 0.0281 | 0.0169 | 0.0393 | 0.0001 | 0.0013 | True |
| Hybrid-Linear+stem | Hybrid-RRF+stem | 0.0159 | 0.0055 | 0.0262 | 0.0037 | 0.0148 | True |
| BM25+stem | BM25 | 0.0192 | 0.0026 | 0.0364 | 0.0283 | 0.0566 | False |
| Hybrid-RRF+stem | Hybrid-RRF | 0.0161 | 0.0033 | 0.0291 | 0.0150 | 0.0450 | True |
| Hybrid-Linear+stem | Hybrid-Linear | 0.0039 | -0.0060 | 0.0140 | 0.4442 | 0.4442 | False |

## Latensi per kueri

| Konfigurasi | median_ms | mean_ms | p95_ms |
|---|---|---|---|
| BM25 | 8.7 | 9.2 | 13.2 |
| BM25+stem | 8.5 | 8.4 | 11.6 |
| Dense | 61.2 | 61.9 | 68.8 |
| Hybrid-RRF | 68.5 | 69.6 | 79.6 |
| Hybrid-RRF+stem | 67.3 | 67.4 | 72.2 |
| Hybrid-Linear | 65.9 | 66.7 | 72.0 |
| Hybrid-Linear+stem | 65.7 | 65.4 | 69.2 |