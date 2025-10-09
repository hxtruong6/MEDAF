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
