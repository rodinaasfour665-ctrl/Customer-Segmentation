# Customer Segmentation Using Machine Learning

**RFM Analysis · K-Means Clustering · Customer Behavior Analytics · Marketing Insights**

An end-to-end customer segmentation project using the **UCI Online Retail** transaction dataset. The project turns raw invoice-line transactions into customer-level behavioral features, evaluates clustering configurations and algorithms, and translates the final segments into actionable marketing recommendations.

## Project Highlights

- **541,909** raw transaction-line records across **8** original columns.
- **392,692** positive sales line records retained after the documented cleaning rules.
- **4,338** customers in the cleaned customer-level modeling dataset.
- RFM (**Recency, Frequency, Monetary**) plus additional purchasing behavior features.
- K-Means experiments covering feature sets, scaling methods, and cluster counts.
- Algorithm benchmarking: **K-Means**, **Agglomerative Clustering**, and **Gaussian Mixture Models**.
- Model validation with **Silhouette**, **Davies–Bouldin**, **Calinski–Harabasz**, sample-based stability testing using **Adjusted Rand Index (ARI)**, and **PCA** visualizations.
- A four-segment marketing interpretation, exported customer labels, segment report, and trained model artifacts.

## Dataset

**Source:** [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)

The dataset contains transactions from a UK-based online retailer spanning **December 2010 to December 2011**. Each row represents a **product line on an invoice**, not a unique customer or necessarily a unique order.

Original fields:

| Column | Description |
| --- | --- |
| `InvoiceNo` | Invoice identifier; identifiers beginning with `C` indicate cancellations |
| `StockCode` | Product identifier |
| `Description` | Product description |
| `Quantity` | Number of items on an invoice line |
| `InvoiceDate` | Invoice timestamp |
| `UnitPrice` | Price per unit |
| `CustomerID` | Customer identifier |
| `Country` | Customer country |

### Data quality and preprocessing

The initial exploration found **135,080 missing `CustomerID` values**, **5,268 exact duplicate rows**, cancellations, and non-positive quantities/prices. The modeling preparation:

1. Preserves a raw copy of the data.
2. Excludes invoice lines without a known `CustomerID` from customer-level modeling; unknown customers are **not** imputed as one artificial customer.
3. Removes exact duplicate rows under a documented working assumption.
4. Separates cancellation/negative-quantity records for potential returns analysis.
5. Uses positive-quantity, positive-price, non-cancelled sales for RFM calculations.
6. Creates `TotalPrice = Quantity × UnitPrice` and aggregates invoice lines by customer.

**Important:** Sales/revenue figures in the project refer to **positive recorded sales for identifiable customers**, not net revenue after refunds, and not company-wide revenue. Negative-quantity records are not all assumed to be confirmed returns.

## Methodology

```text
Raw Online Retail transactions
        ↓
Data understanding, cleaning and validation
        ↓
Customer-level feature engineering (RFM + behavioral features)
        ↓
Distribution, outlier and correlation analysis
        ↓
Log transformation + feature scaling
        ↓
K-Means experiments (feature sets, scalers, number of clusters)
        ↓
Algorithm benchmarking and cluster validation
        ↓
Stability testing + PCA visualization
        ↓
Final customer segments + marketing recommendations
        ↓
CSV reports + saved model and scaler
```

### Customer features

| Feature | Meaning |
| --- | --- |
| `Recency` | Days since the customer's latest observed purchase, relative to a snapshot date |
| `Frequency` | Number of distinct purchase invoices |
| `Monetary` | Total positive sales amount across observed purchases |
| `UniqueProducts` | Number of distinct product codes purchased |
| `TotalItems` | Total quantity purchased |
| `AvgOrderValue` | Monetary amount divided by purchase-invoice count |
| `CustomerTenure` | Days between first purchase and snapshot date |
| `AvgBasketSize` | Total items divided by purchase-invoice count |

The project compares the **RFM** feature set to an **extended behavioral** set. Highly right-skewed features are transformed using `log1p`; experiments compare `StandardScaler` and `RobustScaler`. The final K-Means model uses **log-transformed RFM features with StandardScaler**.

## Model Experiments and Selection

K-Means was evaluated using different feature sets, scalers, and cluster counts (including an Elbow analysis). The benchmark below uses the same scaled RFM representation for each algorithm.

| Algorithm | Clusters (K) | Silhouette ↑ | Davies–Bouldin ↓ |
| --- | ---: | ---: | ---: |
| **K-Means** | **2** | **0.433** | **0.892** |
| Agglomerative | 2 | 0.404 | 0.940 |
| **K-Means** | **4** | **0.337** | **1.011** |
| Agglomerative | 4 | 0.280 | 1.099 |
| GMM | 4 | 0.173 | 1.722 |

**Why K=4?** `K=2` had the best internal clustering scores, but **K-Means with K=4** was selected as a *business-driven trade-off*: four interpretable customer groups provide more actionable marketing differentiation. It is **not** claimed to be the mathematically optimal K by Silhouette.

Additional checks for the four-cluster K-Means solution:

- **Mean ARI: 0.956** across five 80%-training-sample stability experiments, compared with the full-data reference labels.
- **PCA first two components: 93.87%** of the variance in the three-dimensional scaled RFM data, supporting a useful 2D visualization (not proving cluster quality).

These are internal and resampling-based checks, not evidence of future marketing uplift or temporal stability.

## Final Customer Segments

All statistics below are calculated from the project's cleaned customer table. Recency, frequency and monetary values shown are **medians**. Cluster identifiers are model-specific; interpret them through their profiles rather than assuming that the numeric IDs are permanent.

| Cluster | Segment | Customers | Customer Share | Median Recency (days) | Median Frequency | Median Monetary | Positive Sales Share |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | At-Risk | 1,617 | 37.28% | 177 | 1 | 298.40 | 6.33% |
| 1 | Regular | 1,173 | 27.04% | 55 | 4 | 1,341.65 | 23.82% |
| 2 | Recent / Developing | 839 | 19.34% | 17 | 2 | 466.70 | 5.13% |
| 3 | High-Value Loyal | 709 | 16.34% | 8 | 10 | 3,722.82 | 64.72% |

### Recommended marketing actions

| Segment | Interpretation | Proposed action |
| --- | --- | --- |
| **At-Risk** | Infrequent, less recent purchasing activity | Win-back campaigns and personalized reactivation offers |
| **Regular** | Repeat customers with moderate activity and value | Loyalty rewards, relevant cross-selling and retention |
| **Recent / Developing** | Recently active customers with relatively few purchases | Second-purchase incentives and onboarding journeys |
| **High-Value Loyal** | Frequent, recent and high-spending customers | VIP benefits, personalized recognition and priority retention |

**Key insight:** The **High-Value Loyal** segment makes up **16.34% of customers** while contributing **64.72% of positive sales value** in the analyzed customer dataset. The proposed strategies are recommendations; this dataset does not establish their causal impact.

## Tech Stack

- Python
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn (K-Means, Agglomerative Clustering, GMM, PCA, scaling and metrics)
- joblib for model serialization
- Jupyter Notebook / Google Colab
- openpyxl for reading the Excel source

## Repository Structure

Suggested layout when uploading the project:

```text
customer-segmentation/
├── README.md
├── Online_Retail_Customer_Segmentation.ipynb
├── data/
│   └── online+retail.zip
├── outputs/
│   ├── customer_segments.csv
│   └── segment_report.csv
└── models/
    ├── kmeans_customer_model.pkl
    └── customer_scaler.pkl
```

The `.csv` and `.pkl` files are generated when the notebook's export cells are executed. Create the `outputs/` and `models/` folders yourself if using this suggested structure, or adjust the save paths in the notebook. The current notebook's export code writes files to its working directory by default.

## How to Run

1. Clone this repository or download it as a ZIP.
2. Extract `data/online+retail.zip` to obtain **`Online Retail.xlsx`**.
3. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn openpyxl joblib notebook
   ```

4. Open `Online_Retail_Customer_Segmentation.ipynb` in Jupyter Notebook or Google Colab.
5. **Update the dataset path in the notebook's loading cell.** The current Colab-oriented notebook reads:

   ```python
   df = pd.read_excel('/content/Online Retail.xlsx')
   ```

   For local execution, use the path where you extracted the Excel file, for example:

   ```python
   df = pd.read_excel('data/Online Retail.xlsx')
   ```

6. Run all cells in order. If the benchmarking cell was added to your local notebook after Step 20, keep it **before** final model selection/export.
7. Collect the generated CSV and `.pkl` outputs from the notebook working directory, or edit the export cells to write into `outputs/` and `models/`.

### Using the saved model

For future customers, derive RFM features **using the same business definitions and snapshot-date policy** as training. Apply `np.log1p` to `Recency`, `Frequency`, and `Monetary` in that exact column order, then call the **saved scaler's `transform`** followed by the **saved model's `predict`**. Do **not** refit the scaler on a new customer. Segment-name mappings must stay aligned with the fitted model's cluster IDs.

## Limitations and Next Steps

- The data contains no customer **age** or precomputed **spending score**; observed purchase behavior serves as the segmentation basis instead.
- Transactions without `CustomerID` cannot be reliably assigned to customer-level clusters.
- The modeling dataset excludes cancellations and non-positive sales lines from RFM values; a return-aware or net-value model would require additional reconciliation.
- Segment names are descriptive interpretations, not ground-truth labels or validated customer outcomes.
- K-Means prefers compact clusters under Euclidean-distance assumptions. Further checks could test temporal stability, alternative feature weighting, and whether interventions improve retention or repeat purchases.
- The data is historical, so findings should not be interpreted as current consumer behavior.

## Dataset Attribution

Data source: **UCI Machine Learning Repository, Online Retail dataset**. Please consult the [dataset page](https://archive.ics.uci.edu/dataset/352/online+retail) for citation and reuse terms before redistribution.
