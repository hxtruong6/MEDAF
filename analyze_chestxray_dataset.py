#!/usr/bin/env python3
"""
ChestX-ray14 Dataset Analysis Script

This script analyzes the ChestX-ray14 dataset CSV file to provide comprehensive
statistics about the multi-label classification dataset including:
- Total number of images and labels
- Distribution of each pathology label
- Multi-label statistics
- Patient demographics
- Image characteristics
- Label co-occurrence patterns

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

# Set style for better plots
plt.style.use("default")
sns.set_palette("husl")


class ChestXrayAnalyzer:
    def __init__(self, csv_path):
        """
        Initialize the analyzer with the CSV file path

        Args:
            csv_path (str): Path to the Data_Entry_2017.csv file
        """
        self.csv_path = csv_path
        self.df = None
        self.labels = None
        self.label_stats = None

        # ChestX-ray14 pathology labels (14 classes)
        self.pathology_labels = [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
        ]

    def load_data(self):
        """Load and preprocess the CSV data"""
        print("Loading ChestX-ray14 dataset...")

        # Read CSV file
        self.df = pd.read_csv(self.csv_path)

        # Parse the Finding Labels column
        self.df["Finding Labels"] = self.df["Finding Labels"].astype(str)

        # Create binary columns for each pathology
        for label in self.pathology_labels:
            self.df[label] = self.df["Finding Labels"].apply(
                lambda x: 1 if label in x else 0
            )

        # Handle "No Finding" cases
        self.df["No Finding"] = self.df["Finding Labels"].apply(
            lambda x: 1 if x == "No Finding" else 0
        )

        # Parse patient age (remove 'Y' suffix and convert to numeric)
        self.df["Patient Age Numeric"] = (
            self.df["Patient Age"].str.replace("Y", "").astype(int)
        )

        # Parse image dimensions
        self.df["Image Width"] = (
            self.df["OriginalImage[Width,Height]"].str.split(",").str[0].astype(int)
        )
        self.df["Image Height"] = (
            self.df["OriginalImage[Width,Height]"].str.split(",").str[1].astype(int)
        )

        print(f"✅ Dataset loaded successfully!")
        print(f"   Total images: {len(self.df):,}")
        print(f"   Total columns: {len(self.df.columns)}")

    def basic_statistics(self):
        """Print basic dataset statistics"""
        print("\n" + "=" * 60)
        print("📊 BASIC DATASET STATISTICS")
        print("=" * 60)

        print(f"📁 Dataset Information:")
        print(f"   • Total images: {len(self.df):,}")
        print(f"   • Total patients: {self.df['Patient ID'].nunique():,}")
        print(
            f"   • Date range: {self.df['Follow-up #'].min()} - {self.df['Follow-up #'].max()}"
        )

        print(f"\n👥 Patient Demographics:")
        print(
            f"   • Male patients: {len(self.df[self.df['Patient Gender'] == 'M']):,} ({len(self.df[self.df['Patient Gender'] == 'M'])/len(self.df)*100:.1f}%)"
        )
        print(
            f"   • Female patients: {len(self.df[self.df['Patient Gender'] == 'F']):,} ({len(self.df[self.df['Patient Gender'] == 'F'])/len(self.df)*100:.1f}%)"
        )
        print(f"   • Average age: {self.df['Patient Age Numeric'].mean():.1f} years")
        print(
            f"   • Age range: {self.df['Patient Age Numeric'].min()} - {self.df['Patient Age Numeric'].max()} years"
        )

        print(f"\n📸 Image Characteristics:")
        print(f"   • View positions:")
        view_counts = self.df["View Position"].value_counts()
        for view, count in view_counts.items():
            print(f"     - {view}: {count:,} ({count/len(self.df)*100:.1f}%)")

        print(
            f"   • Average image size: {self.df['Image Width'].mean():.0f} x {self.df['Image Height'].mean():.0f}"
        )
        print(
            f"   • Image size range: {self.df['Image Width'].min()}x{self.df['Image Height'].min()} - {self.df['Image Width'].max()}x{self.df['Image Height'].max()}"
        )

    def label_analysis(self):
        """Analyze the distribution of pathology labels"""
        print("\n" + "=" * 60)
        print("🏥 PATHOLOGY LABEL ANALYSIS")
        print("=" * 60)

        # Count occurrences of each label
        label_counts = {}
        for label in self.pathology_labels + ["No Finding"]:
            count = self.df[label].sum()
            percentage = (count / len(self.df)) * 100
            label_counts[label] = {"count": count, "percentage": percentage}

        # Sort by count (descending)
        sorted_labels = sorted(
            label_counts.items(), key=lambda x: x[1]["count"], reverse=True
        )

        print(f"📋 Label Distribution (Total: {len(self.df):,} images):")
        print(f"{'Label':<20} {'Count':<10} {'Percentage':<12} {'Bar'}")
        print("-" * 60)

        for label, stats in sorted_labels:
            count = stats["count"]
            percentage = stats["percentage"]
            bar = "█" * int(percentage / 2)  # Visual bar
            print(f"{label:<20} {count:<10,} {percentage:<11.2f}% {bar}")

        # Store for later use
        self.label_stats = label_counts

        return label_counts

    def multi_label_analysis(self):
        """Analyze multi-label characteristics"""
        print("\n" + "=" * 60)
        print("🔗 MULTI-LABEL ANALYSIS")
        print("=" * 60)

        # Count number of labels per image
        label_columns = self.pathology_labels + ["No Finding"]
        self.df["Num_Labels"] = self.df[label_columns].sum(axis=1)

        # Multi-label statistics
        num_labels_dist = self.df["Num_Labels"].value_counts().sort_index()

        print(f"📊 Labels per Image Distribution:")
        print(f"{'Labels per Image':<20} {'Count':<10} {'Percentage':<12}")
        print("-" * 50)

        for num_labels, count in num_labels_dist.items():
            percentage = (count / len(self.df)) * 100
            print(f"{num_labels:<20} {count:<10,} {percentage:<11.2f}%")

        # Multi-label vs single-label
        single_label = len(self.df[self.df["Num_Labels"] == 1])
        multi_label = len(self.df[self.df["Num_Labels"] > 1])
        no_finding = len(self.df[self.df["No Finding"] == 1])

        print(f"\n📈 Multi-label Statistics:")
        print(
            f"   • Single label images: {single_label:,} ({single_label/len(self.df)*100:.1f}%)"
        )
        print(
            f"   • Multi-label images: {multi_label:,} ({multi_label/len(self.df)*100:.1f}%)"
        )
        print(
            f"   • No finding images: {no_finding:,} ({no_finding/len(self.df)*100:.1f}%)"
        )
        print(f"   • Average labels per image: {self.df['Num_Labels'].mean():.2f}")

        # Most common label combinations
        print(f"\n🔗 Most Common Label Combinations:")
        # Get all unique combinations
        combinations = []
        for idx, row in self.df.iterrows():
            if row["Num_Labels"] > 1:  # Only multi-label cases
                labels = [label for label in self.pathology_labels if row[label] == 1]
                if labels:  # Exclude "No Finding" from combinations
                    combinations.append("|".join(sorted(labels)))

        combo_counts = Counter(combinations)
        top_combinations = combo_counts.most_common(10)

        for i, (combo, count) in enumerate(top_combinations, 1):
            print(f"   {i:2d}. {combo:<30} {count:,} images")

    def label_cooccurrence_analysis(self):
        """Analyze label co-occurrence patterns"""
        print("\n" + "=" * 60)
        print("🔍 LABEL CO-OCCURRENCE ANALYSIS")
        print("=" * 60)

        # Create co-occurrence matrix
        label_columns = self.pathology_labels
        cooccurrence_matrix = np.zeros((len(label_columns), len(label_columns)))

        for i, label1 in enumerate(label_columns):
            for j, label2 in enumerate(label_columns):
                if i != j:
                    # Count images that have both labels
                    both_labels = len(
                        self.df[(self.df[label1] == 1) & (self.df[label2] == 1)]
                    )
                    cooccurrence_matrix[i, j] = both_labels

        # Find strongest co-occurrences
        print("🤝 Strongest Label Co-occurrences:")
        cooccurrence_pairs = []

        for i in range(len(label_columns)):
            for j in range(i + 1, len(label_columns)):
                count = cooccurrence_matrix[i, j]
                if count > 0:
                    cooccurrence_pairs.append(
                        (label_columns[i], label_columns[j], count)
                    )

        # Sort by co-occurrence count
        cooccurrence_pairs.sort(key=lambda x: x[2], reverse=True)

        print(f"{'Label 1':<20} {'Label 2':<20} {'Co-occurrences':<15}")
        print("-" * 60)
        for label1, label2, count in cooccurrence_pairs[:15]:  # Top 15
            print(f"{label1:<20} {label2:<20} {count:<15,}")

    def demographic_analysis(self):
        """Analyze pathology distribution by demographics"""
        print("\n" + "=" * 60)
        print("👥 DEMOGRAPHIC ANALYSIS")
        print("=" * 60)

        # Gender analysis
        print("🚻 Pathology Distribution by Gender:")
        gender_analysis = self.df.groupby("Patient Gender")[self.pathology_labels].sum()

        for gender in ["M", "F"]:
            if gender in gender_analysis.index:
                print(f"\n   {gender}ale patients:")
                gender_counts = gender_analysis.loc[gender].sort_values(ascending=False)
                for label, count in gender_counts.head(10).items():
                    if count > 0:
                        percentage = (
                            count / len(self.df[self.df["Patient Gender"] == gender])
                        ) * 100
                        print(f"     {label:<20} {count:<8,} ({percentage:.1f}%)")

        # Age group analysis
        print(f"\n📅 Pathology Distribution by Age Groups:")
        age_groups = pd.cut(
            self.df["Patient Age Numeric"],
            bins=[0, 30, 50, 70, 100],
            labels=["0-30", "31-50", "51-70", "71+"],
        )
        self.df["Age Group"] = age_groups

        age_analysis = self.df.groupby("Age Group")[self.pathology_labels].sum()

        for age_group in age_analysis.index:
            if pd.notna(age_group):
                print(f"\n   Age {age_group}:")
                age_counts = age_analysis.loc[age_group].sort_values(ascending=False)
                for label, count in age_counts.head(8).items():
                    if count > 0:
                        group_size = len(self.df[self.df["Age Group"] == age_group])
                        percentage = (count / group_size) * 100
                        print(f"     {label:<20} {count:<8,} ({percentage:.1f}%)")

    def create_visualizations(self, save_plots=True):
        """Create comprehensive visualizations"""
        print("\n" + "=" * 60)
        print("📊 CREATING VISUALIZATIONS")
        print("=" * 60)

        # Set up the plotting style
        plt.rcParams["figure.figsize"] = (15, 10)
        plt.rcParams["font.size"] = 10

        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))

        # 1. Label distribution bar plot
        plt.subplot(4, 2, 1)
        label_counts = [
            self.label_stats[label]["count"]
            for label in self.pathology_labels + ["No Finding"]
        ]
        label_names = self.pathology_labels + ["No Finding"]

        bars = plt.bar(
            range(len(label_names)), label_counts, color="skyblue", alpha=0.7
        )
        plt.title("Distribution of Pathology Labels", fontsize=14, fontweight="bold")
        plt.xlabel("Pathology Labels")
        plt.ylabel("Number of Images")
        plt.xticks(range(len(label_names)), label_names, rotation=45, ha="right")

        # Add value labels on bars
        for bar, count in zip(bars, label_counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1000,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 2. Labels per image distribution
        plt.subplot(4, 2, 2)
        num_labels_dist = self.df["Num_Labels"].value_counts().sort_index()
        plt.bar(
            num_labels_dist.index, num_labels_dist.values, color="lightcoral", alpha=0.7
        )
        plt.title("Distribution of Labels per Image", fontsize=14, fontweight="bold")
        plt.xlabel("Number of Labels per Image")
        plt.ylabel("Number of Images")
        plt.xticks(num_labels_dist.index)

        # 3. Gender distribution
        plt.subplot(4, 2, 3)
        gender_counts = self.df["Patient Gender"].value_counts()
        colors = ["lightblue", "lightpink"]
        plt.pie(
            gender_counts.values,
            labels=gender_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        plt.title("Patient Gender Distribution", fontsize=14, fontweight="bold")

        # 4. Age distribution
        plt.subplot(4, 2, 4)
        plt.hist(
            self.df["Patient Age Numeric"],
            bins=20,
            color="lightgreen",
            alpha=0.7,
            edgecolor="black",
        )
        plt.title("Patient Age Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Age (years)")
        plt.ylabel("Number of Patients")

        # 5. View position distribution
        plt.subplot(4, 2, 5)
        view_counts = self.df["View Position"].value_counts()
        plt.bar(view_counts.index, view_counts.values, color="gold", alpha=0.7)
        plt.title("X-ray View Position Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("View Position")
        plt.ylabel("Number of Images")

        # 6. Top pathology combinations
        plt.subplot(4, 2, 6)
        # Get top combinations
        combinations = []
        for idx, row in self.df.iterrows():
            if row["Num_Labels"] > 1:
                labels = [label for label in self.pathology_labels if row[label] == 1]
                if labels:
                    combinations.append("|".join(sorted(labels)))

        combo_counts = Counter(combinations)
        top_combinations = combo_counts.most_common(8)

        if top_combinations:
            combo_names = [
                combo[:20] + "..." if len(combo) > 20 else combo
                for combo, _ in top_combinations
            ]
            combo_values = [count for _, count in top_combinations]

            plt.barh(range(len(combo_names)), combo_values, color="orange", alpha=0.7)
            plt.title("Top Label Combinations", fontsize=14, fontweight="bold")
            plt.xlabel("Number of Images")
            plt.yticks(range(len(combo_names)), combo_names)

        # 7. Age group vs pathology (heatmap)
        plt.subplot(4, 2, 7)
        age_pathology = self.df.groupby("Age Group")[self.pathology_labels].sum()
        if not age_pathology.empty:
            # Normalize by age group size
            age_group_sizes = self.df["Age Group"].value_counts()
            for age_group in age_pathology.index:
                if pd.notna(age_group) and age_group in age_group_sizes:
                    age_pathology.loc[age_group] = (
                        age_pathology.loc[age_group] / age_group_sizes[age_group] * 100
                    )

            sns.heatmap(
                age_pathology.T,
                annot=True,
                fmt=".1f",
                cmap="YlOrRd",
                cbar_kws={"label": "Percentage"},
            )
            plt.title(
                "Pathology Prevalence by Age Group (%)", fontsize=14, fontweight="bold"
            )
            plt.xlabel("Age Group")
            plt.ylabel("Pathology")

        # 8. Image size distribution
        plt.subplot(4, 2, 8)
        plt.scatter(self.df["Image Width"], self.df["Image Height"], alpha=0.5, s=1)
        plt.title("Image Size Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Image Width (pixels)")
        plt.ylabel("Image Height (pixels)")

        plt.tight_layout()

        if save_plots:
            plt.savefig("chestxray_dataset_analysis.png", dpi=300, bbox_inches="tight")
            print("✅ Visualizations saved as 'chestxray_dataset_analysis.png'")

        plt.show()

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "=" * 60)
        print("📋 SUMMARY REPORT")
        print("=" * 60)

        # Calculate key metrics
        total_images = len(self.df)
        total_patients = self.df["Patient ID"].nunique()
        multi_label_images = len(self.df[self.df["Num_Labels"] > 1])
        no_finding_images = len(self.df[self.df["No Finding"] == 1])

        # Most common pathologies
        pathology_counts = {
            label: self.df[label].sum() for label in self.pathology_labels
        }
        most_common = max(pathology_counts, key=pathology_counts.get)
        least_common = min(pathology_counts, key=pathology_counts.get)

        print(f"🎯 Key Findings:")
        print(
            f"   • Dataset contains {total_images:,} chest X-ray images from {total_patients:,} patients"
        )
        print(
            f"   • {multi_label_images:,} images ({multi_label_images/total_images*100:.1f}%) have multiple pathologies"
        )
        print(
            f"   • {no_finding_images:,} images ({no_finding_images/total_images*100:.1f}%) show no abnormalities"
        )
        print(
            f"   • Most common pathology: {most_common} ({pathology_counts[most_common]:,} cases)"
        )
        print(
            f"   • Least common pathology: {least_common} ({pathology_counts[least_common]:,} cases)"
        )
        print(f"   • Average {self.df['Num_Labels'].mean():.2f} labels per image")

        print(f"\n📊 Dataset Characteristics:")
        print(
            f"   • Multi-label classification problem with {len(self.pathology_labels)} pathology classes"
        )
        print(
            f"   • Significant class imbalance (most common vs least common: {pathology_counts[most_common]/pathology_counts[least_common]:.1f}x)"
        )
        print(
            f"   • Patient demographics: {len(self.df[self.df['Patient Gender'] == 'M'])/len(self.df)*100:.1f}% male, {len(self.df[self.df['Patient Gender'] == 'F'])/len(self.df)*100:.1f}% female"
        )
        print(
            f"   • Age range: {self.df['Patient Age Numeric'].min()}-{self.df['Patient Age Numeric'].max()} years (avg: {self.df['Patient Age Numeric'].mean():.1f})"
        )

        print(f"\n💡 Recommendations for Model Training:")
        print(f"   • Use stratified sampling to handle class imbalance")
        print(f"   • Consider focal loss or weighted loss functions")
        print(f"   • Implement data augmentation for rare pathologies")
        print(f"   • Use multi-label evaluation metrics (F1, subset accuracy)")
        print(f"   • Consider patient-level splits to avoid data leakage")

    def run_complete_analysis(self, create_plots=True):
        """Run the complete analysis pipeline"""
        print("🔬 CHESTX-RAY14 DATASET ANALYSIS")
        print("=" * 60)

        # Load data
        self.load_data()

        # Run all analyses
        self.basic_statistics()
        self.label_analysis()
        self.multi_label_analysis()
        self.label_cooccurrence_analysis()
        self.demographic_analysis()

        # Create visualizations
        if create_plots:
            self.create_visualizations()

        # Generate summary
        self.generate_summary_report()

        print(f"\n✅ Analysis complete! Check the generated plots and summary above.")


def main():
    """Main function to run the analysis"""
    # Path to the CSV file
    csv_path = "/home/s2320437/WORK/aidan-medaf/datasets/data/chestxray/NIH/Data_Entry_2017.csv"

    # Create analyzer instance
    analyzer = ChestXrayAnalyzer(csv_path)

    # Run complete analysis
    analyzer.run_complete_analysis(create_plots=True)

    # Optional: Save detailed statistics to CSV
    if analyzer.label_stats:
        stats_df = pd.DataFrame(analyzer.label_stats).T
        stats_df.to_csv("chestxray_label_statistics.csv")
        print("📁 Detailed label statistics saved to 'chestxray_label_statistics.csv'")


if __name__ == "__main__":
    main()
