# ASAP Data Source Documentation

## Overview
This document outlines the sources and acquisition methods for the Crop Mask and Crop Calendar data used in this project. All data is sourced from the **Anomaly Hotspots of Agricultural Production (ASAP)** system, developed by the Joint Research Council (JRC) of the European Commission.

**Source Website:** [https://agricultural-production-hotspots.ec.europa.eu/download.php](https://agricultural-production-hotspots.ec.europa.eu/download.php)

**Source Website:** [https://agricultural-production-hotspots.ec.europa.eu/download.php](https://agricultural-production-hotspots.ec.europa.eu/download.php)

---

## 0. Global Data Configuration
All data collection should aim to cover the following period where possible:
*   **Target Start Year:** 2007
*   **Target End Year:** 2025 (or most recent available)
*   **Target Countries:** Kenya, Ethiopia, Somalia

*Note: Some datasets (e.g., VIIRS, Sentinel-2) may have later start dates due to satellite launch times. In such cases, the earliest available data should be used.*

---

## 1. Crop Mask Data (Global)
The crop mask dataset provides a global grid indicating the fraction of area covered by crops.

*   **File Name:** `asap_mask_crop_v04.tif`
*   **Description:**
    *   **Content:** Global Crop Mask (area fraction).
    *   **Resolution:** ~1/4 square kilometer (0.00446 degrees).
    *   **Units:** Pixel values represent the percentage (0-100) of the pixel area covered by cropland.
    *   **Version:** v04 (Updated 2023-12-01).
    *   **Location on Site:** Under "REFERENCE DATA" -> "Land Cover Masks".
*   **Download Link:** [Direct Link](https://agricultural-production-hotspots.ec.europa.eu/files/asap_mask_crop_v04.tif)
*   **Acquisition Method:**
    *   Downloaded via `curl` command.
    *   Command: `curl -o asap_mask_crop_v04.tif https://agricultural-production-hotspots.ec.europa.eu/files/asap_mask_crop_v04.tif`

---

## 2. Crop Calendar Data
The crop calendar datasets define the timing of key crop development stages (planting, growth, harvest) based on FAO data and remote sensing phenology.

### A. Sub-national Level (GAUL1)
*   **File Name:** `crop_calendar_gaul1.zip`
*   **Description:**
    *   **Content:** Crop calendar data aggregated at the first administrative level (GAUL1).
    *   **Fields:** Contains start/end dekads for planting, growing season, and harvesting.
    *   **Location on Site:** Under "REFERENCE DATA" -> "Crop Calendars".
*   **Download Link:** [Direct Link](https://agricultural-production-hotspots.ec.europa.eu/files/crop_calendar_gaul1.zip)
*   **Acquisition Method:**
    *   Downloaded via `curl` command.
    *   Command: `curl -o crop_calendar_gaul1.zip https://agricultural-production-hotspots.ec.europa.eu/files/crop_calendar_gaul1.zip`

### B. National Level (GAUL0)
*   **File Name:** `crop_calendar_gaul0.zip`
*   **Description:**
    *   **Content:** Crop calendar data aggregated at the national level (GAUL0).
    *   **Location on Site:** Under "REFERENCE DATA" -> "Crop Calendars".
*   **Download Link:** [Direct Link](https://agricultural-production-hotspots.ec.europa.eu/files/crop_calendar_gaul0.zip)
*   **Acquisition Method:**
    *   Downloaded via `curl` command.
    *   Command: `curl -o crop_calendar_gaul0.zip https://agricultural-production-hotspots.ec.europa.eu/files/crop_calendar_gaul0.zip`

---

## 3. RATIN Cross-Border Trade (XBT) Data
**Source:** [RATIN (Regional Agricultural Trade Intelligence Network)](https://ratin.net)

*   **Status:** **Login/Subscription Required**
*   **Access Findings:**
    *   Real-time and historical cross-border trade flow data is available via the **RATIN Trade Analytics** platform.
    *   Access requires user registration and likely a subscription specific to trade analytics.
    *   Public reports (e.g., Grain Watch) are listed but currently appear to be placeholders on the site.
    *   **Registration Page:** [https://ratin.net/ratinapp/subscribers/register.php](https://ratin.net/ratinapp/subscribers/register.php)
*   **Recommendation:** User needs to register/subscribe to access the full datasets.

---

## 4. Night Time Light Data

### A. Colorado School of Mines (EOG) - VIIRS Nighttime Lights (Recommended)
This is the standard source for research-grade nighttime light data.
*   **Source:** [EOG VIIRS Nighttime Lights (VNL)](https://eogdata.mines.edu/products/vnl/)
*   **Data Type:** VIIRS Day/Night Band (DNB) - Monthly and Annual composites.
*   **Access Method:**
    *   **Direct Download:** Available from the website.
    *   **API:** REST API available for programmatic access.
    *   **Python Wrapper:** `nightpy` library or custom scripts using `requests`.
*   **Authentication:** Requires a free account registration (Client ID & Secret).

### B. NASA Black Marble
High-quality, daily and monthly products (VNP46).
*   **Source:** [NASA Black Marble](https://blackmarble.gsfc.nasa.gov/)
*   **Access Method:** Use the `blackmarblepy` library for easy downloading.
    *   **Install:** `pip install blackmarblepy`
*   **Authentication:** Requires a NASA Earthdata Login (Bear Token).

### C. Google Earth Engine (GEE)
Best for cloud-based analysis but can also export data.
*   **Catalog:** [VIIRS Nighttime Day/Night Band](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMCFG)
*   **Python API:** `earthengine-api`
*   **Note:** Requires a Google Cloud project and GEE account.
