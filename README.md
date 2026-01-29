# Room Detector - ESP32 CSI-based Indoor Positioning

## Overview
An innovative indoor positioning and room detection system using ESP32 microcontrollers and Channel State Information (CSI) technology. This project leverages WiFi signal analysis to detect and classify different rooms or areas in indoor environments without requiring GPS or external positioning systems.

## Features
- **ESP32 CSI Technology**: Utilize Channel State Information for precise indoor positioning
- **Room Classification**: Automatically detect and classify different rooms
- **Real-time Monitoring**: Live tracking and position updates
- **WiFi-based Positioning**: No GPS required for indoor navigation
- **Machine Learning Integration**: AI-powered location classification
- **Low-cost Hardware**: Affordable ESP32-based solution

## Technology Stack
- **Hardware**: ESP32 microcontrollers
- **Wireless**: WiFi CSI (Channel State Information)
- **Programming**: C++ (Arduino/ESP-IDF), Python
- **Machine Learning**: Scikit-learn, TensorFlow
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Real-time dashboards

## Installation
1. Clone the repository including ESP32-CSI-Tool submodule
2. Set up ESP32 development environment:
   ```bash
   # Install ESP-IDF or Arduino IDE
   # Configure ESP32 toolchain
   ```
3. Install Python dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```
4. Flash ESP32 firmware with CSI capabilities

## Hardware Setup
- **ESP32 Boards**: Multiple ESP32 devices for triangulation
- **WiFi Access Points**: Existing WiFi infrastructure
- **Power Supply**: USB or battery power for ESP32 devices
- **Positioning**: Strategic placement of ESP32 devices in rooms

## Project Structure
- `ESP32-CSI-Tool/` - ESP32 firmware and CSI data collection tools
- `active_ap/` - Active access point configuration and management
- Machine learning models for room classification
- Data processing and visualization scripts

## How It Works

### 1. CSI Data Collection
- ESP32 devices monitor WiFi signals in the environment
- Extract Channel State Information from WiFi frames
- Collect amplitude and phase information from multiple antennas
- Record signal characteristics unique to each location

### 2. Signal Processing
- **Preprocessing**: Filter and normalize CSI data
- **Feature Extraction**: Extract relevant signal features
- **Noise Reduction**: Remove environmental interference
- **Calibration**: Account for hardware variations

### 3. Machine Learning Classification
- **Training Data**: Collect CSI fingerprints for each room
- **Feature Engineering**: Create discriminative features
- **Model Training**: Train classifiers for room detection
- **Real-time Prediction**: Classify current location based on CSI

## Key Features

### Channel State Information (CSI)
- **Amplitude Data**: Signal strength variations
- **Phase Information**: Signal phase characteristics
- **Multi-antenna Data**: MIMO channel information
- **Frequency Response**: Channel response across subcarriers

### Room Detection Capabilities
- **Indoor Positioning**: Accurate room-level localization
- **Movement Tracking**: Track movement between rooms
- **Occupancy Detection**: Detect presence in rooms
- **Environmental Mapping**: Create indoor signal maps

## Applications
- **Smart Home Automation**: Room-specific device control
- **Security Systems**: Intrusion detection and monitoring
- **Healthcare**: Patient monitoring in hospitals
- **Retail Analytics**: Customer movement analysis
- **Office Management**: Space utilization tracking
- **IoT Integration**: Location-aware IoT devices

## Configuration
- **ESP32 Settings**: WiFi configuration, CSI parameters
- **Access Point Setup**: Configure existing or dedicated APs
- **Calibration**: Room-specific calibration procedures
- **ML Models**: Training parameters and model selection

## Data Collection Process
1. **Environment Setup**: Position ESP32 devices strategically
2. **Calibration Phase**: Collect training data in each room
3. **Feature Extraction**: Process CSI data for ML training
4. **Model Training**: Train room classification models
5. **Deployment**: Real-time room detection system

## Performance Metrics
- **Accuracy**: Room classification accuracy (typically 85-95%)
- **Response Time**: Real-time detection latency
- **Range**: Effective detection range per ESP32
- **Reliability**: System uptime and error rates

## Advantages over Traditional Systems
- **No GPS Required**: Works in GPS-denied environments
- **Low Cost**: Inexpensive ESP32 hardware
- **No Infrastructure**: Uses existing WiFi networks
- **High Precision**: Room-level accuracy
- **Real-time**: Immediate position updates
- **Scalable**: Easy to add more rooms or areas

## Technical Specifications
- **Frequency**: 2.4 GHz WiFi (802.11n/ac)
- **Hardware**: ESP32 with CSI capabilities
- **Range**: Typical 10-30 meters per device
- **Accuracy**: Room-level positioning (1-3 meters)
- **Power**: Low power consumption for battery operation

## Development Tools
- **ESP-IDF**: Espressif IoT Development Framework
- **Arduino IDE**: Alternative development environment
- **CSI Tools**: Specialized CSI data collection utilities
- **Python Scripts**: Data processing and ML training

## Contributing
1. Fork the repository
2. Experiment with different CSI features
3. Improve machine learning algorithms
4. Test in various indoor environments
5. Submit pull request

## Requirements
- ESP32 development boards (multiple units recommended)
- WiFi network infrastructure
- Python 3.7+ for data processing
- Basic electronics and programming knowledge

## Limitations
- **WiFi Dependency**: Requires active WiFi environment
- **Environmental Sensitivity**: Affected by furniture changes
- **Calibration**: Requires initial setup and calibration
- **2.4 GHz Only**: Limited to 2.4 GHz WiFi bands

## Future Enhancements
- **5 GHz Support**: Extend to 5 GHz WiFi bands
- **Mobile App**: Smartphone interface for monitoring
- **Cloud Integration**: Remote monitoring and analytics
- **Multi-floor Support**: 3D positioning capabilities
- **Enhanced ML**: Deep learning for better accuracy

## License
MIT License

## Research References
- Channel State Information applications
- Indoor positioning system research
- WiFi-based localization studies
- ESP32 CSI implementation guides

## Acknowledgments
- ESP32-CSI-Tool contributors
- Indoor positioning research community
- ESP32 and WiFi technology developers
