## Overview

VERSION: 0.4.8

API Client for accessing data from Google Maps, Apple Maps, OpenStreetMaps and PropertyRadar maps.

## Prerequisites

Make sure you have the following installed:

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git (optional for the development environment)

### Miniconda Installation

To install Miniconda, follow the instructions for your operating system:

#### Linux

1. Download the Miniconda installer for Linux:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

2. Run the installer:

```
bash Miniconda3-latest-Linux-x86_64.sh
```

3. Follow the prompts to complete the installation.

#### Windows

1. Download the Miniconda installer for Windows from the [official website](https://docs.conda.io/en/latest/miniconda.html).

2. Run the installer and follow the prompts to complete the installation.

#### macOS

1. Download the Miniconda installer for macOS:

```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
```

2. Run the installer:
```
bash Miniconda3-latest-MacOSX-x86_64.sh
```

3. Follow the prompts to complete the installation.

## Production Environment Installation

To install the package in a new Conda environment for production use, follow these steps:

1. Create a new Conda environment:

```
conda create --name map_tile_client python=3.9
```

2. Activate the environment:
```
conda activate map_tile_client
```

3. Install the package directly from the GitHub repository:
```
pip install git+ssh://git@github.com/nrahnemoon/MapTileClient.git
```

## Development Environment Installation

To set up the development environment, follow these steps:

1. Clone the repository:
```
git clone git@github.com:nrahnemoon/MapTileClient.git
```

2. Change to the repository directory:
```
cd MapTileClient
```

3. Create a new Conda environment:
```
conda create --name map_tile_client python=3.9
```

4. Activate the environment:
```
conda activate map_tile_client
```

5. Install the package in development mode with dev extras:
```
poetry install
```

## Release Notes

*0.1.0* (2023-05-06) Initial version

*0.1.1* (2023-05-06) Update cache dir

*0.1.2* (2023-05-07) Add from_map static initializer to BaseMap

*0.1.3* (2023-05-07) Bugfix BaseMap not defined

*0.2.0* (2023-05-07) Add get_mono_map to AppleMapsStandardMap

*0.3.0* (2023-05-07) Flake8 and black + import fixes

*0.4.0* (2023-05-27) Update property radar parcel tile URL

*0.4.1* (2023-05-27) Add get lat lon and get parcel px to PropertyRadarParcelMap

*0.4.2* (2023-05-27) Bugfix in get latlon and contour px

*0.4.3* (2023-05-27) Fix imports

*0.4.4* (2023-05-27) Update Google Maps Standard Map to get roof mono image

*0.4.5* (2023-05-27) Mono map roof for google amps

*0.4.6* (2023-05-27) Invert mono map for google standard map

*0.4.7* (2023-05-29) Fix multithread

*0.4.8* (2023-05-29) Multithread bugfix
