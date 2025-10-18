아래는 자연스러운 연구 제안서 스타일로 깔끔하게 다듬어 번역한 영어본입니다. 한국어 원문의 구조·의미는 유지하면서 문장 흐름만 학술적으로 매끈하게 손봤어요.

⸻

Eastern Africa Food Price Forecasting Proposal

Building a Time-Series Prediction Model Integrating Climate Change, Conflict, and Market Drivers

Current date: October 18, 2025

1. Project Overview

This project aims to develop advanced machine-learning and deep-learning models to forecast food price volatility across Eastern Africa. We will holistically analyze the combined effects of climate change, local conflict, and socioeconomic factors on food prices to build a more accurate and reliable forecasting system. The system is intended to improve early-warning accuracy for food crises and support evidence-based policymaking that strengthens regional food security.

1.1. Background and Rationale

Eastern Africa is among the regions most vulnerable to climate change. Recurrent extreme events—droughts and floods—pose severe threats to agricultural productivity.[4] Such climate variability exacerbates supply instability, which in turn drives food price spikes and social unrest. As seen in Nigeria, rising temperatures can trigger “heatflation,” directly pushing up food prices.[7]

Analyses of ACLED (Armed Conflict Location & Event Data Project) further indicate a meaningful positive feedback loop between food price increases and conflict incidence across Africa: price surges can fuel conflict, and conflict disrupts markets and supply chains, which then elevates prices further.[5] In this complex, compounding risk environment, organizations such as the World Food Programme (WFP) monitor market functionality, food availability, and supply chains to maximize the effectiveness of humanitarian responses. [Web-9]

Many prior forecasting studies focus on single drivers or coarse national aggregates and thus fail to capture local heterogeneity. Our project addresses these limitations by using micro-level (market-level) data and integrating multi-domain features—climate, conflict, and socioeconomic indicators. A key emphasis is on combining deep time-series models with spatial analytics to learn complex, nonlinear, spatio-temporal interactions among variables and thereby improve predictive precision.

1.2. Project Objectives
	•	High-resolution dataset construction: Build a spatio-temporal dataset for Eastern Africa that integrates market-level food prices with climate, conflict, infrastructure, and socioeconomic data.
	•	Multivariate forecasting models: Establish statistical time-series baselines (SARIMA, VAR), and develop ML/DL models (e.g., LSTM, GRU) and ensemble methods (e.g., XGBoost) to forecast food price inflation.
	•	Driver attribution: Use model interpretability (XAI) to quantify which features—climate, conflict, market access, etc.—most strongly affect food price dynamics by region and time.
	•	Early-warning prototype: Propose a conceptual design for an early-warning system that flags impending price spikes in specific locales based on model forecasts.

1.3. Expected Contributions
	•	Policy impact: Provide scientific evidence for governments and international agencies to act preemptively. For example, when drought risk is forecast, authorities can plan prepositioning and distribution that account for fragile local transport networks.
	•	Scholarly impact: Advance empirical understanding of climate–conflict–price interactions at micro scales and contribute methodologically by applying modern ML/DL techniques to this domain.
	•	Technical impact: Deliver an automated, reproducible data pipeline that ingests, cleans, and merges heterogeneous data sources, along with MLflow-based experiment management to ensure reproducibility and scalability.

⸻

2. Data Pipeline and Sources

Project success hinges on accurately fusing multi-dimensional data in space and time. The pipeline automates collection, cleaning, and integration to produce modeling-ready datasets. Below are the target variables, predictors, and collection plans.

Figure 1. End-to-end methodology from data ingestion to forecasting.[2]

2.1. Target Variable

Data	Metric	Granularity	Source	Description
WFP Food Price Data	Retail prices of key staples by market	Market level, monthly	WFP VAM DataViz [Web-8], HDX [Web-10]	Monthly prices for major staples (e.g., maize, sorghum, beans) in specific markets in Eastern Africa. The ALPS (Alert for Price Spikes) indicator [Web-14] will support anomaly detection.

2.2. Predictor Variables

Predictors span five categories—climate, conflict, agriculture/market, geospatial/infrastructure, and macroeconomy—and will be fetched via APIs or direct downloads and spatially joined to market locations using GIS.

2.2.1. Climate Data
	•	Precipitation: Monthly totals, anomalies vs. long-term means, SPI (CHIRPS, TRMM, TAMSAT)[4]. We will model drought/flood impacts on production and logistics, computing departures from climatology as in Raleigh et al. (2015).[5]
	•	Temperature: Monthly land surface temperature (LST), frequency of extreme heat/cold (MODIS LST, Landsat, CRU Timeseries)[3]. For “heatflation,” we will compute days above critical thresholds (e.g., 32 °C) as crop stress indicators.[1]
	•	Drought indices: VCI, NDVI, PDSI (MODIS, AVHRR)[4], enabling lagged effects of agricultural drought on yields and prices.
	•	Extreme events: Tropical cyclone tracks/intensity (IBTrACS, NOAA)[6]. Following Bao et al. (2022)[6], we will encode whether a cyclone passed within X km of a market and its peak wind speed to capture supply-chain shocks.

2.2.2. Conflict Data
	•	Conflict incidence: Monthly counts within X km of each market, fatalities, and event types (e.g., battles, violence against civilians) from ACLED [Web-12]. As in Raleigh et al. (2015)[5], we will quantify effects on market access, transport costs, and agricultural activity. These variables can also support instrumented strategies to address price–conflict endogeneity.

2.2.3. Crop & Market Data
	•	Crop characteristics: Storage periods for key crops (FAO, national statistics). Following Bao et al. (2022)[6], perishables with shorter storage may be more shock-sensitive.
	•	Market functionality: Availability, supply-chain status, inventories (WFP Market Functionality Index, MFI) [Web-8], used to assess buffering capacity against volatility.

2.2.4. Geospatial/Infrastructure Data
	•	Market access: Distance to primary roads, road density, distance to neighboring markets (OpenStreetMap, World Bank). Proxies for transport costs and price transmission; we will test whether fragile networks amplify climate shocks.
	•	Population density: Gridded population (WorldPop, GPW) as a demand-side pressure indicator and to study sensitivity in dense areas.

2.2.5. Macroeconomic Data
	•	External factors: Global oil prices, exchange rates, global food price indices (IMF, World Bank)[7], to capture pass-through where import dependence is high.

2.3. Collection Pipeline

The pipeline is modularized under src/data_pipeline, with scripts for download and initial cleaning. It is automated for reproducibility and scale.

data_pipeline/
├── wfp_downloader.py       # WFP VAM API – price data
├── climate_downloader.py   # Google Earth Engine, Copernicus APIs – climate data
├── acled_downloader.py     # ACLED API – conflict data
├── osm_parser.py           # OpenStreetMap – road network features
└── run_pipeline.sh         # Orchestrates the full pipeline

Raw data are versioned using tools such as DVC and stored in source-specific formats (e.g., GeoTIFF, CSV).

⸻

3. Data Preprocessing

Before modeling, data undergo multiple preprocessing steps to improve quality, harmonize formats, and engineer features. Scripts live under src/preprocessing_data.

3.1. Target Variable: Food Price Inflation

WFP market-level monthly prices are transformed into log inflation:
	•	Missing values: For short gaps, use linear interpolation or moving averages; exclude markets/items with prolonged gaps.
	•	Seasonal decomposition: Use STL or classical decomposition to separate seasonality, trend, and residuals, so models learn exogenous shocks rather than harvest-cycle seasonality. [Web-17]
	•	Stationarity & inflation: Apply log transform and differencing to stabilize variance and remove trends. Define:
\text{Inflation\_rate}(t) = \log(\text{Price}t) - \log(\text{Price}{t-1})
This models changes rather than levels for easier interpretation.[6]

3.2. Predictor Engineering
	•	Climate lags: Create 1–6-month lags for precipitation, temperature, and drought indices to capture delayed impacts (cf. Bao et al., 2022).[6]
	•	Climate anomalies: Compute standardized departures (Z-scores) for monthly precipitation/temperature relative to long-run means and standard deviations, following Raleigh et al. (2015).[5]
	•	Extreme-event features: For buffers around each market (e.g., 50/100 km), encode binary occurrence or intensity (e.g., max wind speed; runs of ≥ 35 °C for ≥ 3 days).
	•	Conflict aggregation: Aggregate ACLED point events monthly within market buffers; include counts, fatalities, and disaggregate by type (battles, VAC, protests).
	•	Scaling: Standardize or min-max scale numeric features to stabilize DL training.

3.3. Dataset Merging

Geospatial fusion is the core of the final panel.
	•	Reference geometry: Market latitude/longitude from WFP serve as spatial anchors.
	•	Raster sampling: Extract pixel values (or buffered means) from rasters (climate, population) at market locations.
	•	Vector joins: Use spatial joins/proximity for vector data (conflict points, road lines), e.g., distance to nearest conflict, total road length within 50 km.
	•	Final panel: Merge all variables by (market_id, year-month) to form a spatio-temporal panel for modeling. All steps are scripted for reproducibility.

⸻

4. Modeling Architecture

We will progress from classical statistical baselines to modern DL architectures to capture the complexity of price dynamics. Development occurs under src/model, with iterative complexity and benchmarking.

4.1. Baselines: Statistical Time Series
	•	SARIMA: Treat each market as an independent series and model AR, MA, and seasonal components. Use insights from decomposition to set (p,d,q)(P,D,Q)_s.
	•	VAR: Jointly model multiple markets to capture cross-market spillovers and propagation of shocks via a system of equations.

4.2. ML/DL Models

Following the comparative approach validated by Khan et al. (2024)[1], we will evaluate multiple models:
	•	Tree ensembles: XGBoost / LightGBM for strong accuracy and speed, with feature importance for interpretability; Random Forest for robust baselines.
	•	Deep time series: LSTM/GRU for long-term dependencies (e.g., prolonged drought or sustained conflict effects).[1]

Proposed DL stack:

Input: [timesteps, num_features]
 → LSTM/GRU (128, return_sequences=True)
 → Dropout (0.2)
 → LSTM/GRU (64)
 → Dropout (0.2)
 → Dense (32, relu)
 → Output (1, linear)   # regression

Here, timesteps (e.g., 12 months) controls look-back horizon; num_features is the total predictor count. Dropout mitigates overfitting.

4.3. Advanced Approaches
	•	Spatio-temporal models: Explicitly account for spatial autocorrelation (e.g., SAR) observed in Eastern Africa agricultural analyses.[Web-3]
	•	Spatio-Temporal Graph Neural Networks (ST-GNN): Build a market graph (nodes = markets; edges = road connectivity/proximity) to learn dynamic propagation of shocks.
	•	Quantile regression: Instead of mean effects, estimate heterogeneous effects across the distribution (e.g., 90th-percentile spikes) as in Erdogan et al. (2024).[7]
	•	Causality & endogeneity: Because prices and conflict are mutually endogenous (Raleigh et al., 2015)[5], consider:
	•	Simultaneous equations: Jointly estimate price and conflict equations to capture feedback.
	•	Instrumental variables: Use instruments that affect conflict but not prices directly (e.g., election cycles, political marginalization indices) to identify exogenous conflict variation.

4.4. Evaluation

Adopt the comprehensive framework of Khan et al. (2024)[1].
	•	Metrics: RMSE, MAE, MAPE.
	•	Validation: Time-series cross-validation via walk-forward or rolling origin to prevent leakage.
	•	Model selection: Choose finalists on accuracy, stability, and interpretability.

⸻

5. Project Management and Execution Plan

We emphasize modular code structure and rigorous experiment tracking to maximize reproducibility and efficiency.

5.1. Code Structure

project_root/
├── data/
│   ├── raw/             # raw sources
│   └── processed/       # cleaned/engineered
├── notebooks/           # EDA & prototyping
├── src/
│   ├── data_pipeline/   # ingestion scripts
│   ├── preprocessing/   # cleaning & feature engineering
│   ├── models/          # model definitions
│   ├── train.py         # training & evaluation runner
│   └── predict.py       # inference runner
├── tests/               # unit/integration tests
└── mlflow_runs/         # MLflow artifacts

5.2. Experiment Management (MLflow)

Track and compare:
	•	Parameters: hyperparameters (learning rate, layer counts), feature sets, window sizes.
	•	Metrics: RMSE/MAE/MAPE.
	•	Artifacts: trained models, plots (predicted vs. actual), feature importance.
	•	Source code: Git commit hashes for full reproducibility.
This enables efficient comparison across many experiments to select optimal models and settings.

5.3. Phased Roadmap

Phase	Key Activities	Duration	Deliverables
1	Data ingestion & pipeline setup	4 weeks	Automated collectors; versioned raw datasets
2	EDA & preprocessing	4 weeks	Visualization report; preprocessing pipeline; modeling-ready dataset
3	Baselines & ML/DL modeling	6 weeks	Trained SARIMA/VAR/XGBoost/LSTM; MLflow results
4	Advanced modeling & analysis	5 weeks	Spatio-temporal models; quantile analysis; final models & performance report
5	Final report & prototype	3 weeks	Final report; early-warning system concept; presentation deck


⸻

6. References
	1.	Khan, F., Liou, Y. A., Spöck, G., Wang, X., Ali, S., & Abbas, M. (2024). Assessing the impacts of temperature extremes on agriculture yield and projecting future extremes using machine learning and deep learning approaches with CMIP6 data. IJAEOG, 132, 104071. https://doi.org/10.1016/j.jag.2024.104071
	2.	Garajeh, M. K., et al. (2023). An integrated approach of remote sensing and geospatial analysis for modeling and predicting the impacts of climate change on food security. Scientific Reports, 13(1), 1057. https://doi.org/10.1038/s41598-023-28244-5
	3.	Haile, M. G., et al. (2017). Impact of Climate Change, Weather Extremes, and Price Risk on Global Food Supply. Food Security. https://doi.org/10.1007/s12571-017-0686-z
	4.	Bhaga, T. D., et al. (2020). Impacts of Climate Variability and Drought on Surface Water Resources in Sub-Saharan Africa Using Remote Sensing: A Review. Remote Sensing, 12(24), 4184. https://doi.org/10.3390/rs12244184
	5.	Raleigh, C., Choi, H. J., & Kniveton, D. (2015). The devil is in the details: An investigation of the relationships between conflict, food price and climate across Africa. Global Environmental Change, 32, 187-199. http://dx.doi.org/10.1016/j.gloenvcha.2015.03.005
	6.	Bao, X., Sun, P., & Li, J. (2023). The impacts of tropical storms on food prices: Evidence from China. AJAE, 105(2), 576-596. https://doi.org/10.1111/ajae.12330
	7.	Erdogan, S., Kartal, M. T., & Pata, U. K. (2024). Does Climate Change Cause an Upsurge in Food Prices? Foods, 13(1), 154. https://doi.org/10.3390/foods13010154
	8.	Lloyd, S. J., et al. (2018). A global-level model of the potential impacts of climate change on child stunting via income and food price in 2030. EHP, 126(9), 097007. https://doi.org/10.1289/EHP2916
	9.	Iimi, A., You, L., & Wood-Sichra, U. (2015). Agriculture production and transport infrastructure in East Africa: An application of spatial autoregression. World Bank Policy Research Working Paper. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2613210
	10.	WFP VAM (2025). Eastern Africa – Prices. [Web-8]
	11.	World Food Programme (2025). Market analysis. [Web-9]
	12.	HDX (2025). Global Food Prices Database (WFP). [Web-10]
	13.	ACLED (2025). ACLED data. [Web-12]
	14.	World Food Programme (2024). Forecasting trends in food security with real-time data. Nature Communications Earth & Environment. https://www.nature.com/articles/s43247-024-01698-9
	15.	Preset.io (2025). Time Series Forecasting: A Complete Guide. [Web-17]