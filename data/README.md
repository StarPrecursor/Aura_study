# Access data

## Data Resources API

- Binance: https://github.com/binance/binance-public-data

## Binance data

Folder: /binance

Scripts:

- 1021_get_complete_data.py: download binance data with given API
  - collect data from 2021-09-01 to 2022-09-01
- 1021_merge.py: merge all data with same symbol
  - rerun to merge with updated data
- 1109_update:
  - collect data from 2022-09-01 to 2022-11-08

## Tardis data

Folder: /tardis

Scripts:

- example/fundingRateDownload.py
  - run with required arguments to update data
