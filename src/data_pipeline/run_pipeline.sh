#!/bin/bash

# Eastern Africa Food Price Forecasting Data Pipeline
# Orchestrates the download of all data sources outlined in agents.md

set -e  # Exit on any error

# Configuration
START_YEAR=2019
END_YEAR=2024
LOG_DIR="logs"
DATA_DIR="data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create necessary directories
echo -e "${BLUE}Setting up directories...${NC}"
mkdir -p $LOG_DIR
mkdir -p $DATA_DIR/raw/{wfp,acled,climate,osm,macro,geospatial}
mkdir -p $DATA_DIR/processed

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to run downloader with error handling
run_downloader() {
    local script_name=$1
    local description=$2
    local log_file="$LOG_DIR/${script_name%.py}.log"

    echo -e "${BLUE}Starting: $description${NC}"
    log "Starting $description" >> $log_file

    if python3 src/data_pipeline/$script_name >> $log_file 2>&1; then
        echo -e "${GREEN}✓ Completed: $description${NC}"
        log "Completed successfully: $description" >> $log_file
        return 0
    else
        echo -e "${RED}✗ Failed: $description${NC}"
        echo -e "${YELLOW}Check log: $log_file${NC}"
        log "Failed: $description" >> $log_file
        return 1
    fi
}

# Function to install required packages
install_dependencies() {
    echo -e "${BLUE}Installing Python dependencies...${NC}"

    # Check if requirements.txt exists, if not create basic one
    if [ ! -f "requirements.txt" ]; then
        cat > requirements.txt << EOF
requests>=2.31.0
pandas>=2.0.0
geopandas>=0.13.0
numpy>=1.24.0
xarray>=2023.1.0
rasterio>=1.3.0
shapely>=2.0.0
overpy>=0.7
wbdata>=0.3.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
folium>=0.14.0
plotly>=5.14.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
EOF
        echo "Created basic requirements.txt"
    fi

    # Install packages
    pip3 install -r requirements.txt
}

# Main pipeline execution
main() {
    echo -e "${GREEN}=== Eastern Africa Food Price Forecasting Data Pipeline ===${NC}"
    echo -e "${BLUE}Start time: $(date)${NC}"
    echo ""

    # Install dependencies
    install_dependencies

    # Track success/failure
    declare -a success_list
    declare -a failure_list

    # 1. WFP Food Price Data
    if run_downloader "wfp_downloader.py" "WFP Food Price Data"; then
        success_list+=("WFP Food Price Data")
    else
        failure_list+=("WFP Food Price Data")
    fi

    # 2. ACLED Conflict Data
    if run_downloader "acled_downloader.py" "ACLED Conflict Data"; then
        success_list+=("ACLED Conflict Data")
    else
        failure_list+=("ACLED Conflict Data")
    fi

    # 3. Climate Data
    if run_downloader "climate_downloader.py" "Climate Data (CHIRPS, MODIS, etc.)"; then
        success_list+=("Climate Data")
    else
        failure_list+=("Climate Data")
    fi

    # 4. OpenStreetMap Infrastructure
    if run_downloader "osm_parser.py" "OpenStreetMap Infrastructure Data"; then
        success_list+=("OpenStreetMap Infrastructure")
    else
        failure_list+=("OpenStreetMap Infrastructure")
    fi

    # 5. Macroeconomic Data
    if run_downloader "macro_downloader.py" "Macroeconomic Data (World Bank, IMF)"; then
        success_list+=("Macroeconomic Data")
    else
        failure_list+=("Macroeconomic Data")
    fi

    # 6. Geospatial and Population Data
    if run_downloader "geospatial_downloader.py" "Geospatial and Population Data"; then
        success_list+=("Geospatial and Population Data")
    else
        failure_list+=("Geospatial and Population Data")
    fi

    # Generate summary report
    echo ""
    echo -e "${GREEN}=== PIPELINE SUMMARY ===${NC}"
    echo -e "${BLUE}End time: $(date)${NC}"
    echo ""

    if [ ${#success_list[@]} -gt 0 ]; then
        echo -e "${GREEN}✓ Successfully downloaded (${#success_list[@]} sources):${NC}"
        for item in "${success_list[@]}"; do
            echo -e "  ${GREEN}✓${NC} $item"
        done
    fi

    echo ""

    if [ ${#failure_list[@]} -gt 0 ]; then
        echo -e "${RED}✗ Failed downloads (${#failure_list[@]} sources):${NC}"
        for item in "${failure_list[@]}"; do
            echo -e "  ${RED}✗${NC} $item"
        done
    else
        echo -e "${GREEN}All data sources downloaded successfully!${NC}"
    fi

    # Save summary to file
    {
        echo "Eastern Africa Food Price Forecasting - Data Pipeline Summary"
        echo "============================================================="
        echo "Execution Date: $(date)"
        echo "Time Period: $START_YEAR - $END_YEAR"
        echo ""
        echo "Successfully Downloaded (${#success_list[@]} sources):"
        for item in "${success_list[@]}"; do
            echo "✓ $item"
        done
        echo ""
        echo "Failed Downloads (${#failure_list[@]} sources):"
        for item in "${failure_list[@]}"; do
            echo "✗ $item"
        done
        echo ""
        echo "Data Structure:"
        find data -type f -name "*.csv" -o -name "*.geojson" -o -name "*.json" | head -20
        echo ""
        echo "Log files located in: $LOG_DIR/"
    } > $LOG_DIR/pipeline_summary.txt

    echo ""
    echo -e "${BLUE}Detailed summary saved to: $LOG_DIR/pipeline_summary.txt${NC}"

    # Show data directory structure
    echo ""
    echo -e "${BLUE}Data directory structure:${NC}"
    tree data/ 2>/dev/null || find data -type f | head -20

    # Return appropriate exit code
    if [ ${#failure_list[@]} -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# Help function
show_help() {
    echo "Eastern Africa Food Price Forecasting Data Pipeline"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -y, --year     Set end year (default: $END_YEAR)"
    echo "  -s, --start    Set start year (default: $START_YEAR)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run with default years"
    echo "  $0 -s 2020 -y 2024   # Run for 2020-2024"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -y|--year)
            END_YEAR="$2"
            shift 2
            ;;
        -s|--start)
            START_YEAR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main pipeline
main