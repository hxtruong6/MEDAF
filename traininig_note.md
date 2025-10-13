# Stragegy 1: only known labels without new labels

## without any weighting and threshold optimization

Overall Performance:
   Subset Accuracy:  0.4317 (43.17%)
   Hamming Accuracy: 0.8924 (89.24%)
   Precision:        0.3818
   Recall:           0.0913
   F1-Score:         0.1365
   Average Loss:     0.7675
   Threshold Used:   0.5

🏷️  Per-Class Performance:
   Class           Precision  Recall     F1-Score  
   --------------------------------------------------

   Atelectasis     0.4437     0.0811     0.1371
   Cardiomegaly    0.4941     0.1174     0.1897
   Effusion        0.4976     0.3083     0.3807
   Infiltration    0.3856     0.0913     0.1477
   Mass            0.4532     0.0537     0.0961
   Nodule          0.3766     0.0557     0.0971
   Pneumonia       0.0000     0.0000     0.0000
   Pneumothorax    0.4040     0.0229     0.0434

🏆 Best Class:  Effusion (F1=0.3807)
📉 Worst Class: Pneumonia (F1=0.0000)

## with class weighting and threshold optimization

 TODO:

# Stragegy 2: trainning data with new labels but not declear in training data

📊 Overall Performance:
   Subset Accuracy:  0.2709 (27.09%)
   Hamming Accuracy: 0.8502 (85.02%)
   Precision:        0.2854
   Recall:           0.3078
   F1-Score:         0.2826
   Average Loss:     0.6557

🏷️  Per-Class Performance (Optimal Thresholds):
   Class           Threshold  Precision  Recall     F1-Score  
   -----------------------------------------------------------------

   Atelectasis     0.200      0.3276     0.2494     0.2832
   Cardiomegaly    0.050      0.3459     0.3232     0.3342
   Effusion        0.200      0.4326     0.5074     0.4670
   Infiltration    0.150      0.3089     0.7026     0.4291
   Mass            0.050      0.2611     0.2863     0.2731
   Nodule          0.050      0.2315     0.1819     0.2038
   Pneumonia       0.500      0.0000     0.0000     0.0000
   Pneumothorax    0.050      0.3752     0.2118     0.2708

🏆 Best Class:  Effusion (F1=0.4670)
📉 Worst Class: Pneumonia (F1=0.0000)

### weighted loss

📊 Overall Performance:
   Subset Accuracy:  0.2224 (22.24%)
   Hamming Accuracy: 0.7897 (78.97%)
   Precision:        0.2269
   Recall:           0.4914
   F1-Score:         0.3086
   Average Loss:     0.7209

🏷️  Per-Class Performance (Optimal Thresholds):
   Class           Threshold  Precision  Recall     F1-Score  
   -----------------------------------------------------------------

   Atelectasis     0.600      0.2403     0.5989     0.3430
   Cardiomegaly    0.900      0.2518     0.4799     0.3303
   Effusion        0.750      0.3703     0.6420     0.4697
   Infiltration    0.650      0.3224     0.7540     0.4517
   Mass            0.600      0.1672     0.4869     0.2490
   Nodule          0.750      0.1513     0.3619     0.2134
   Pneumonia       0.750      0.0414     0.0684     0.0516
   Pneumothorax    0.700      0.2702     0.5395     0.3601

🏆 Best Class:  Effusion (F1=0.4697)
📉 Worst Class: Pneumonia (F1=0.0516)

💡 Model Assessment:
   🔶 Model shows good improvement with optimal thresholds
   📈 Consider: expert configuration tuning, advanced loss functions
✅ Evaluation completed successfully

--------------------- 30 epoch ---------------------
📊 Overall Performance:
   Subset Accuracy:  0.2497 (24.97%)
   Hamming Accuracy: 0.8071 (80.71%)
   Precision:        0.2483
   Recall:           0.4398
   F1-Score:         0.3099
   Average Loss:     0.3994

🏷️  Per-Class Performance (Optimal Thresholds):
   Class           Threshold  Precision  Recall     F1-Score  
   -----------------------------------------------------------------

   Atelectasis     0.550      0.2439     0.5702     0.3417
   Cardiomegaly    0.750      0.3266     0.3477     0.3368
   Effusion        0.700      0.3839     0.6473     0.4820
   Infiltration    0.500      0.3392     0.5848     0.4293
   Mass            0.250      0.2245     0.3520     0.2741
   Nodule          0.400      0.1781     0.2335     0.2021
   Pneumonia       0.050      0.0326     0.2009     0.0561
   Pneumothorax    0.150      0.2577     0.5822     0.3572

🏆 Best Class:  Effusion (F1=0.4820)
📉 Worst Class: Pneumonia (F1=0.0561)

📈 AUC Performance Metrics:
   Macro AUC:    0.7266
   Micro AUC:    0.7911
   Weighted AUC: 0.7254
   Valid Classes: 8/8

🏷️  Per-Class AUC Scores:
   Class           AUC Score  Performance
   ---------------------------------------------

   Atelectasis     0.7204     Fair
   Cardiomegaly    0.8488     Good
   Effusion        0.7911     Fair
   Infiltration    0.6498     Poor
   Mass            0.7401     Fair
   Nodule          0.6780     Poor
   Pneumonia       0.6026     Poor
   Pneumothorax    0.7822     Fair

# Traning all labels

Testing DataLoader 0: 100%|███████████████████████████████████████████████████████████████████| 800/800 [02:20<00:00,  5.71it/s]
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Runningstage.testing metric ┃        DataLoader 0         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│    test/auc_Atelectasis     │     0.6836501359939575      │
│    test/auc_Cardiomegaly    │     0.8152111172676086      │
│   test/auc_Consolidation    │      0.681266725063324      │
│       test/auc_Edema        │     0.7905856966972351      │
│      test/auc_Effusion      │     0.7376695871353149      │
│     test/auc_Emphysema      │     0.7509016990661621      │
│      test/auc_Fibrosis      │     0.7406420707702637      │
│       test/auc_Hernia       │     0.8283058404922485      │
│    test/auc_Infiltration    │     0.6637631058692932      │
│        test/auc_Mass        │     0.6427419185638428      │
│       test/auc_Nodule       │     0.6350199580192566      │
│ test/auc_Pleural_Thickening │     0.6808257699012756      │
│     test/auc_Pneumonia      │     0.6391271948814392      │
│    test/auc_Pneumothorax    │     0.7743315696716309      │
│       test/macro_auc        │     0.7188601659683209      │
│       test/micro_auc        │     0.7298349738121033      │
│      test/weighted_auc      │     0.7037743330001831      │
└─────────────────────────────┴─────────────────────────────┘
2025-10-13 15:25:16 - core.lightning_trainer - INFO - Testing completed successfully
2025-10-13 15:25:16 - __main__ - INFO - [2/2] Testing novelty detection...
2025-10-13 15:25:16 - core.lightning_trainer - INFO - Evaluating novelty detection from checkpoint: checkpoints/medaf_lightning/medaf-lightning-epoch=10-val_loss=0.0000.ckpt
/home/s2320437/miniconda3/envs/research_medaf_aidan/lib/python3.12/site-packages/lightning_fabric/utilities/cloud_io.py:51: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See <https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models> for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  return torch.load(f, map_location=map_location)  # type: ignore[arg-type]
Filtered to 86524 training images
Loaded 86524 samples with 14 classes
Class weights calculated:
  Class 0: 9.3968
  Class 1: 49.6320
  Class 2: 8.9823
  Class 3: 5.3039
  Class 4: 20.5652
  Class 5: 17.4794
  Class 6: 97.8223
  Class 7: 31.8158
  Class 8: 29.3831
  Class 9: 62.0542
  Class 10: 59.2258
  Class 11: 68.0967
  Class 12: 37.5314
  Class 13: 588.9394
Filtered to 25596 test images
Loaded 25596 samples with 14 classes
Loaded 13739 all samples for novelty detection
Threshold calibrated: 1.2215 (FPR target: 5.0%)

======================================================================
🔍 NOVELTY DETECTION EVALUATION RESULTS
======================================================================

📊 Overall Novelty Detection Performance:
   AUROC:              0.5353
   Detection Accuracy: 0.6449 (64.49%)
   Precision:          0.1299
   Recall:             0.0029
   F1-Score:           0.0057

🎯 Detection Threshold: 1.2215
   Known samples:   25596
   Unknown samples: 13739

💡 Performance Assessment: Needs Improvement ⚠️
======================================================================

2025-10-13 15:32:47 - __main__ - INFO - ✅ Comprehensive evaluation completed successfully!

======================================================================
📋 COMPREHENSIVE EVALUATION SUMMARY
======================================================================

✅ Standard Classification Results:
  test/macro_auc: 0.7189
  test/micro_auc: 0.7298
  test/weighted_auc: 0.7038
  test/auc_Atelectasis: 0.6837
  test/auc_Cardiomegaly: 0.8152
  test/auc_Effusion: 0.7377
  test/auc_Infiltration: 0.6638
  test/auc_Mass: 0.6427
  test/auc_Nodule: 0.6350
  test/auc_Pneumonia: 0.6391
  test/auc_Pneumothorax: 0.7743
  test/auc_Consolidation: 0.6813
  test/auc_Edema: 0.7906
  test/auc_Emphysema: 0.7509
  test/auc_Fibrosis: 0.7406
  test/auc_Pleural_Thickening: 0.6808
  test/auc_Hernia: 0.8283

🔍 Novelty Detection Results:
  AUROC: 0.5353
  Detection Accuracy: 0.6449
  F1-Score: 0.0057
======================================================================

(research_medaf_aidan) s2320437@spcc-a100g05 ~/WORK/aidan-medaf ±main⚡ »
