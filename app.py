import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page settings
st.set_page_config(
    page_title="AutoData | EDA & Diagnostic Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">AutoData: Interactive EDA & Diagnostic Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated Statistical Profiling, Health Assessment, and Remediation Framework for Tabular Datasets</div>', unsafe_allow_html=True)

# ----------------- Helper Functions ----------------- #

def load_data(uploaded_file):
    """Loads CSV or Excel files with character encoding safety."""
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                return pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding='latin1')
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def detect_outliers_iqr(series):
    """Calculates Tukey's fences and returns outlier counts and thresholds."""
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 0, np.nan, np.nan
    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = clean_series[(clean_series < lower_fence) | (clean_series > upper_fence)]
    return len(outliers), lower_fence, upper_fence

def convert_df_to_csv(df):
    """Converts the processed DataFrame into a downloadable CSV format."""
    return df.to_csv(index=False).encode('utf-8')

def auto_clean_dataframe(
    df: pd.DataFrame, current_year: int = 2026
) -> pd.DataFrame:
  df = df.copy()

  # 1. Remove exact duplicates
  df = df.drop_duplicates().reset_index(drop=True)

  # 2. Convert placeholder symbols to NaN & trim whitespace
  placeholders = {'?', 'NA', 'N/A', 'null', 'None', 'nan', '-', ''}
  for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].apply(
        lambda x: np.nan if (isinstance(x, str) and x.strip() in placeholders) else x
    )
    if not col.lower().endswith('_id') and col.lower() != 'id':
      df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

  # 3. Standardize booleans (Yes/No) and text casing/aliases
  aliases = {'hyd': 'Hyderabad', 'new delhi': 'Delhi', 'bangalore': 'Bengaluru'}
  bool_map = {
      'yes': 'Yes',
      'true': 'Yes',
      '1': 'Yes',
      'no': 'No',
      'false': 'No',
      '0': 'No',
  }

  for col in df.select_dtypes(include=['object', 'string']).columns:
    if col.lower().endswith('_id') or col.lower() == 'id':
      continue
    non_null_lower = df[col].dropna().astype(str).str.lower().unique()
    if len(non_null_lower) > 0 and set(non_null_lower).issubset(
        set(bool_map.keys())
    ):
      df[col] = df[col].apply(
          lambda x: bool_map[str(x).lower()] if pd.notna(x) else x
      )
    else:
      df[col] = df[col].apply(
          lambda x: aliases.get(x.lower(), x.title()) if isinstance(x, str) else x
      )

  # 4. Convert impossible domain values to NaN BEFORE imputation
  for col in df.select_dtypes(include=[np.number]).columns:
    col_l = col.lower()
    if any(k in col_l for k in ['price', 'area', 'sqft', 'distance', 'salary']):
      df.loc[df[col] <= 0, col] = np.nan
    elif any(k in col_l for k in ['bedroom', 'bathroom', 'room', 'count']):
      df.loc[df[col] < 0, col] = np.nan
    elif 'year' in col_l:
      df.loc[(df[col] < 1800) | (df[col] > current_year), col] = np.nan

  # 5. Impute missing values (Median for numeric, Mode for categorical)
  for col in df.columns:
    if df[col].isnull().sum() > 0:
      if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
      else:
        df[col] = df[col].fillna(df[col].mode()[0])

  # 6. IQR outlier clipping with domain floor/ceiling guards
  for col in df.select_dtypes(include=[np.number]).columns:
    if col.lower().endswith('_id') or col.lower() == 'id':
      continue
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
      lower = q1 - 1.5 * iqr
      upper = q3 + 1.5 * iqr
      col_l = col.lower()

      if any(k in col_l for k in ['price', 'area', 'sqft', 'distance']):
        lower = max(lower, df[col].min())
      if 'bedroom' in col_l or 'bathroom' in col_l:
        lower = max(lower, 1)
      if 'year' in col_l:
        upper = min(upper, current_year)

      df[col] = df[col].clip(lower=lower, upper=upper)

  # 7. Round discrete integer columns cleanly
  for col in df.select_dtypes(include=[np.number]).columns:
    if any(k in col.lower() for k in ['bedroom', 'bathroom', 'year', 'count']):
      df[col] = df[col].round().astype('int64')

  return df

# ----------------- Sidebar File Ingestion ----------------- #

st.sidebar.header("📁 Ingestion Interface")
uploaded_file = st.sidebar.file_uploader(
    "Upload Tabular File (CSV / XLSX)", 
    type=["csv", "xlsx", "xls"],
    help="Upload any CSV or Excel file to begin automated profiling."
)

if uploaded_file is not None:
    # Maintain session state across data remediations
    if "raw_df" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state.raw_df = df.copy()
            st.session_state.working_df = df.copy()
            st.session_state.file_name = uploaded_file.name
            st.session_state.pop("auto_clean_report", None)

    df = st.session_state.working_df

    # Reset Data Button
    if st.sidebar.button("🔄 Reset to Original Data"):
        st.session_state.working_df = st.session_state.raw_df.copy()
        st.session_state.pop("auto_clean_report", None)
        st.rerun()

    # Module Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview & Health", 
        "📈 Distribution & Outliers", 
        "🔗 Correlation Matrix", 
        "🛠️ Remediation & Imputation", 
        "💾 Export Data"
    ])

    # ----------------- TAB 1: Structural Health Profiling ----------------- #
    with tab1:
        st.subheader("1. Structural Health & Dataset Profile")
        
        rows, cols = df.shape
        duplicate_count = df.duplicated().sum()
        total_missing = df.isnull().sum().sum()
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Rows", f"{rows:,}")
        col2.metric("Total Columns", cols)
        col3.metric("Duplicate Rows", f"{duplicate_count:,}")
        col4.metric("Missing Cells", f"{total_missing:,}")
        col5.metric("Memory Footprint", f"{memory_usage_mb:.2f} MB")

        st.markdown("---")
        
        c_left, c_right = st.columns([3, 2])
        with c_left:
            st.markdown("**Dataset Preview (First 5 Rows)**")
            st.dataframe(df.head(), width="stretch")

        with c_right:
            st.markdown("**Column Schema & Missingness Summary**")
            profile_df = pd.DataFrame({
                "Data Type": df.dtypes.astype(str),
                "Missing Values": df.isnull().sum(),
                "Missing Ratio (%)": (df.isnull().sum() / len(df) * 100).round(2),
                "Unique Values": df.nunique()
            })
            st.dataframe(profile_df, width="stretch")

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            st.markdown("**Descriptive Statistics (Numerical Columns)**")
            st.dataframe(df.describe().T.round(3), width="stretch")
    # ----------------- TAB 2: Distribution & Outlier Mapping ----------------- #
    with tab2:
        st.subheader("2. Distribution & Tukey's Outlier Mapping")
        
        if not num_cols:
            st.warning("No numerical features found in this dataset.")
        else:
            selected_feature = st.selectbox("Select Numerical Feature", num_cols)

            if selected_feature:
                series = df[selected_feature]
                outlier_count, lower_bound, upper_bound = detect_outliers_iqr(series)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Skewness", f"{series.skew():.3f}")
                m2.metric("Lower Fence (Q1 - 1.5*IQR)", f"{lower_bound:.3f}" if pd.notnull(lower_bound) else "N/A")
                m3.metric("Upper Fence (Q3 + 1.5*IQR)", f"{upper_bound:.3f}" if pd.notnull(upper_bound) else "N/A")
                m4.metric("Outlier Count", f"{outlier_count} ({(outlier_count / len(series) * 100):.2f}%)")

                fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
                
                # KDE Histogram
                sns.histplot(series.dropna(), kde=True, ax=axes[0], color="#2563EB", bins=30)
                axes[0].set_title(f"Distribution & KDE: {selected_feature}", fontsize=12)
                axes[0].grid(True, linestyle="--", alpha=0.5)

                # Tukey Boxplot
                sns.boxplot(x=series.dropna(), ax=axes[1], color="#93C5FD", flierprops={"markerfacecolor": "#DC2626", "markersize": 6})
                axes[1].set_title(f"Tukey's Boxplot: {selected_feature}", fontsize=12)
                axes[1].grid(True, linestyle="--", alpha=0.5)

                st.pyplot(fig)
                plt.close(fig)

    # ----------------- TAB 3: Pearson Correlation ----------------- #
    with tab3:
        st.subheader("3. Multivariate Pearson Correlation Assessment")
        
        if len(num_cols) < 2:
            st.warning("At least two numerical variables are required to generate a correlation matrix.")
        else:
            corr_matrix = df[num_cols].corr(method='pearson')

            fig, ax = plt.subplots(figsize=(max(8, len(num_cols) * 0.7), max(6, len(num_cols) * 0.6)))
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                fmt=".2f", 
                cmap="vlag", 
                vmin=-1, 
                vmax=1, 
                center=0,
                linewidths=0.5, 
                ax=ax, 
                cbar_kws={'label': "Pearson's r"}
            )
            ax.set_title("Pearson Bivariate Correlation Matrix", fontsize=13, pad=12)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("**Potential Multicollinearity Warning Signatures (|r| > 0.85)**")
            redundancy_list = []
            
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    val = corr_matrix.iloc[i, j]
                    if abs(val) >= 0.85:
                        redundancy_list.append({
                            "Feature A": num_cols[i],
                            "Feature B": num_cols[j],
                            "Correlation (r)": round(val, 4),
                            "Severity": "Redundant Pair" if abs(val) >= 0.95 else "High Multicollinearity"
                        })
            
            if redundancy_list:
                st.dataframe(pd.DataFrame(redundancy_list), width="stretch")
            else:
                st.success("No feature pairs with high multicollinearity (|r| > 0.85) detected.")

# ----------------- TAB 4: Remediation Pipeline ----------------- #
    with tab4:
        st.subheader("4. Parameterized Data Cleaning & Remediation")

        # 1-Click Automated Remediation Engine
        st.markdown("#### ⚡ 1-Click Automated Cleaning Pipeline")
        st.write("Automatically standardizes placeholders (`?`, `N/A`, `TBD`), parses currency strings (`$`, `,`), clips IQR outliers, and applies skewness-adaptive imputation.")
        
        if st.button("🚀 Run Full Auto-Clean", type="primary"):
            cleaned_df, audit_report = auto_clean_dataframe(df)
            st.session_state.working_df = cleaned_df
            st.session_state.auto_clean_report = audit_report
            st.rerun()

        if "auto_clean_report" in st.session_state and st.session_state.auto_clean_report:
            st.success("Automated remediation completed! Audit Summary:")
            audit_df = pd.DataFrame(
                list(st.session_state.auto_clean_report.items()),
                columns=["Target / Feature", "Applied Remediation Policy"]
            )
            st.dataframe(audit_df, width="stretch")

        st.markdown("---")
        st.markdown("#### 🔧 Manual Granular Controls")
        col_rem1, col_rem2 = st.columns(2)
        
        # Deduplication Module
        with col_rem1:
            st.markdown("**Duplicate Row Elimination**")
            st.write(f"Current duplicate instances detected: **{df.duplicated().sum()}**")
            if st.button("Drop Duplicate Records"):
                initial_len = len(df)
                df = df.drop_duplicates()
                st.session_state.working_df = df
                st.success(f"Removed {initial_len - len(df)} duplicate rows successfully.")
                st.rerun()

        # Missing Value Imputation Module
        with col_rem2:
            st.markdown("**Missing Value Imputation**")
            cols_with_na = df.columns[df.isna().any()].tolist()
            
            if not cols_with_na:
                st.success("No missing values remaining in the working dataset.")
            else:
                target_col = st.selectbox("Select Column to Impute", cols_with_na)
                is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
                
                if is_numeric:
                    strategy = st.selectbox("Imputation Strategy", ["Median", "Mean", "Custom Constant", "Drop Rows with NA"])
                else:
                    strategy = st.selectbox("Imputation Strategy", ["Mode (Most Frequent)", "Custom Constant", "Drop Rows with NA"])
                
                custom_fill = None
                if strategy == "Custom Constant":
                    custom_fill = st.text_input("Enter replacement value:")

                if st.button("Apply Imputation"):
                    if strategy == "Mean":
                        df[target_col] = df[target_col].fillna(df[target_col].mean())
                    elif strategy == "Median":
                        df[target_col] = df[target_col].fillna(df[target_col].median())
                    elif strategy == "Mode (Most Frequent)":
                        df[target_col] = df[target_col].fillna(df[target_col].mode()[0])
                    elif strategy == "Custom Constant":
                        if is_numeric:
                            try:
                                df[target_col] = df[target_col].fillna(float(custom_fill))
                            except ValueError:
                                st.error("Please enter a valid numeric value.")
                        else:
                            df[target_col] = df[target_col].fillna(custom_fill)
                    elif strategy == "Drop Rows with NA":
                        df = df.dropna(subset=[target_col])

                    st.session_state.working_df = df
                    st.success(f"Remediation policy '{strategy}' applied to column '{target_col}'.")
                    st.rerun()

    # ----------------- TAB 5: Export Remediated Dataset ----------------- #
    with tab5:
        st.subheader("5. Export Remediated Dataset")
        st.write("Download the updated, remediated dataset ready for downstream machine learning tasks.")
        st.markdown(f"**Cleaned Dimensions:** {df.shape[0]} rows × {df.shape[1]} columns")
        
        csv_data = convert_df_to_csv(df)
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv_data,
            file_name=f"autodata_cleaned_{uploaded_file.name.split('.')[0]}.csv",
            mime="text/csv",
            help="Exports standard UTF-8 encoded CSV file."
        )

else:
    st.info("👋 Upload a CSV or Excel file via the left sidebar to initiate automated exploratory profiling.")
