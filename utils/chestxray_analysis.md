# ChestX-ray14 Dataset Analysis

```bash
❯ python analyze_chestxray_dataset.py

❯ python analyze_chestxray_dataset.py
🔬 CHESTX-RAY14 DATASET ANALYSIS
============================================================
Loading ChestX-ray14 dataset...
        Image Index          Finding Labels  Follow-up #  Patient ID Patient Age Patient Gender View Position  OriginalImage[Width  Height]  OriginalImagePixelSpacing[x  ...  Pneumonia  Pneumothorax  Consolidation  Edema  Emphysema  Fibrosis  Pleural_Thickening  Hernia  No Finding  Patient Age Numeric
0  00000001_000.png            Cardiomegaly            0           1        058Y              M            PA                 2682     2749                        0.143  ...          0             0              0      0          0         0                   0       0           0                   58
1  00000001_001.png  Cardiomegaly|Emphysema            1           1        058Y              M            PA                 2894     2729                        0.143  ...          0             0              0      0          1         0                   0       0           0                   58
2  00000001_002.png   Cardiomegaly|Effusion            2           1        058Y              M            PA                 2500     2048                        0.168  ...          0             0              0      0          0         0                   0       0           0                   58
3  00000002_000.png              No Finding            0           2        081Y              M            PA                 2500     2048                        0.171  ...          0             0              0      0          0         0                   0       0           1                   81
4  00000003_000.png                  Hernia            0           3        081Y              F            PA                 2582     2991                        0.143  ...          0             0              0      0          0         0                   0       1           0                   81

[5 rows x 27 columns]
✅ Dataset loaded successfully!
   Total images: 112,120
   Total columns: 29

============================================================
📊 BASIC DATASET STATISTICS
============================================================
📁 Dataset Information:
   • Total images: 112,120
   • Total patients: 30,805
   • Date range: 0 - 183

👥 Patient Demographics:
   • Male patients: 63,340 (56.5%)
   • Female patients: 48,780 (43.5%)
   • Average age: 46.9 years
   • Age range: 1 - 414 years

📸 Image Characteristics:
   • View positions:
     - PA: 67,310 (60.0%)
     - AP: 44,810 (40.0%)
   • Average image size: 2646 x 2486
   • Image size range: 1143x966 - 3827x4715

============================================================
🏥 PATHOLOGY LABEL ANALYSIS
============================================================
📋 Label Distribution (Total: 112,120 images):
Label                Count      Percentage   Bar
------------------------------------------------------------
No Finding           60,412     53.88      % ██████████████████████████
Infiltration         19,870     17.72      % ████████
Effusion             13,307     11.87      % █████
Atelectasis          11,535     10.29      % █████
Nodule               6,323      5.64       % ██
Mass                 5,746      5.12       % ██
Pneumothorax         5,298      4.73       % ██
Consolidation        4,667      4.16       % ██
Pleural_Thickening   3,385      3.02       % █
Cardiomegaly         2,772      2.47       % █
Emphysema            2,516      2.24       % █
Edema                2,303      2.05       % █
Fibrosis             1,686      1.50       %
Pneumonia            1,353      1.21       %
Hernia               227        0.20       %

============================================================
🔗 MULTI-LABEL ANALYSIS
============================================================
📊 Labels per Image Distribution:
Labels per Image     Count      Percentage
--------------------------------------------------
1                    91,385     81.51      %
2                    14,292     12.75      %
3                    4,829      4.31       %
4                    1,233      1.10       %
5                    298        0.27       %
6                    64         0.06       %
7                    16         0.01       %
8                    1          0.00       %
9                    2          0.00       %

📈 Multi-label Statistics:
   • Single label images: 91,385 (81.5%)
   • Multi-label images: 20,735 (18.5%)
   • No finding images: 60,412 (53.9%)
   • Average labels per image: 1.26

🔗 Most Common Label Combinations:
    1. Effusion|Infiltration          1,602 images
    2. Atelectasis|Infiltration       1,356 images
    3. Atelectasis|Effusion           1,167 images
    4. Infiltration|Nodule            829 images
    5. Atelectasis|Effusion|Infiltration 740 images
    6. Cardiomegaly|Effusion          483 images
    7. Consolidation|Infiltration     442 images
    8. Infiltration|Mass              420 images
    9. Effusion|Pneumothorax          405 images
   10. Effusion|Mass                  401 images

============================================================
🔍 LABEL CO-OCCURRENCE ANALYSIS
============================================================
🤝 Strongest Label Co-occurrences:
Label 1              Label 2              Co-occurrences
------------------------------------------------------------
Effusion             Infiltration         3,990.0
Atelectasis          Effusion             3,269.0
Atelectasis          Infiltration         3,259.0
Infiltration         Nodule               1,544.0
Effusion             Consolidation        1,287.0
Effusion             Mass                 1,244.0
Atelectasis          Consolidation        1,222.0
Infiltration         Consolidation        1,220.0
Infiltration         Mass                 1,151.0
Cardiomegaly         Effusion             1,060.0
Effusion             Pneumothorax         995.0
Infiltration         Edema                979.0
Infiltration         Pneumothorax         943.0
Effusion             Nodule               909.0
Mass                 Nodule               894.0

============================================================
👥 DEMOGRAPHIC ANALYSIS
============================================================
🚻 Pathology Distribution by Gender:

   Male patients:
     Infiltration         11,412   (18.0%)
     Effusion             7,427    (11.7%)
     Atelectasis          6,892    (10.9%)
     Nodule               3,683    (5.8%)
     Mass                 3,510    (5.5%)
     Pneumothorax         2,715    (4.3%)
     Consolidation        2,666    (4.2%)
     Pleural_Thickening   2,042    (3.2%)
     Emphysema            1,610    (2.5%)
     Cardiomegaly         1,306    (2.1%)

   Fale patients:
     Infiltration         8,458    (17.3%)
     Effusion             5,880    (12.1%)
     Atelectasis          4,643    (9.5%)
     Nodule               2,640    (5.4%)
     Pneumothorax         2,583    (5.3%)
     Mass                 2,236    (4.6%)
     Consolidation        2,001    (4.1%)
     Cardiomegaly         1,466    (3.0%)
     Pleural_Thickening   1,343    (2.8%)
     Edema                1,099    (2.3%)

📅 Pathology Distribution by Age Groups:

   Age 0-30:
     Infiltration         4,175    (19.6%)
     Effusion             1,984    (9.3%)
     Atelectasis          1,489    (7.0%)
     Pneumothorax         1,154    (5.4%)
     Mass                 965      (4.5%)
     Consolidation        931      (4.4%)
     Nodule               884      (4.2%)
     Cardiomegaly         534      (2.5%)

   Age 31-50:
     Infiltration         6,748    (17.4%)
     Effusion             4,108    (10.6%)
     Atelectasis          3,597    (9.3%)
     Nodule               2,047    (5.3%)
     Mass                 1,790    (4.6%)
     Pneumothorax         1,693    (4.4%)
     Consolidation        1,525    (3.9%)
     Pleural_Thickening   949      (2.4%)

   Age 51-70:
     Infiltration         7,754    (17.1%)
     Effusion             6,072    (13.4%)
     Atelectasis          5,497    (12.1%)
     Nodule               3,010    (6.6%)
     Mass                 2,645    (5.8%)
     Pneumothorax         2,131    (4.7%)
     Consolidation        1,883    (4.2%)
     Pleural_Thickening   1,664    (3.7%)

   Age 71+:
     Infiltration         1,190    (17.6%)
     Effusion             1,142    (16.9%)
     Atelectasis          951      (14.1%)
     Nodule               382      (5.7%)
     Mass                 343      (5.1%)
     Consolidation        328      (4.9%)
     Pneumothorax         319      (4.7%)
     Pleural_Thickening   279      (4.1%)

============================================================
📋 SUMMARY REPORT
============================================================
🎯 Key Findings:
   • Dataset contains 112,120 chest X-ray images from 30,805 patients
   • 20,735 images (18.5%) have multiple pathologies
   • 60,412 images (53.9%) show no abnormalities
   • Most common pathology: Infiltration (19,870 cases)
   • Least common pathology: Hernia (227 cases)
   • Average 1.26 labels per image

📊 Dataset Characteristics:
   • Multi-label classification problem with 14 pathology classes
   • Significant class imbalance (most common vs least common: 87.5x)
   • Patient demographics: 56.5% male, 43.5% female
   • Age range: 1-414 years (avg: 46.9)
```
