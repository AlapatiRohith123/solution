# NASA Mars Spectrometry EGA-MS Dataset License and Attribution

## Source Dataset
This frozen benchmark subset is derived from:
- **Dataset**: NASA Mars Spectrometry: Detect Chemical Compounds on Mars (EGA-MS)
- **Challenge Host**: DrivenData (2022) in partnership with NASA
- **Repository**: Registry of Open Data on AWS (`https://registry.opendata.aws/nasa-ega/`)
- **Original Source**: NASA Goddard Space Flight Center (GSFC) & NASA Johnson Space Center (JSC) Sample Analysis at Mars (SAM) science team; archived in the NASA Planetary Data System (`https://pds.nasa.gov/`).

## License and Allowed Use
This dataset is published under the **Creative Commons Zero 1.0 Universal (CC0 1.0)** public domain dedication in accordance with the NASA Science Mission Directorate Open-Source Science Policy. There are no restrictions on the access, use, modification, or distribution of this data.

### Attribution Statement
As requested by NASA and the dataset providers:
> "NASA provided support for the development of SAM. The datasets for these challenges were provided by NASA Goddard Space Flight Center and NASA Johnson Space Center. They have been collected by the Sample Analysis at Mars (SAM) scientists and specifically processed for these challenges with the help of the scientists from NASA: Doug Archer, Charles Malespin, Caroline Freissinet, Stephanie Getty, Luoth Chou, Eric Lyness, and Victoria Da Poian, and the DrivenData team. Data from all SAM experiments are archived in the Planetary Data System (pds.nasa.gov)."

## Subset Selection and Benchmark Transformations
The bundled task input is a curated, frozen subset derived from the NASA EGA-MS corpus:
1. **Sample Selection**: 31 samples total (21 training samples, 10 validation samples) representing both commercial laboratory instruments and the SAM testbed flight spare.
2. **Features**: Mass-to-charge ($m/z$) spectral time-temperature profiles packaged as CSV files within `train_features.zip` and `val_features.zip`.
3. **Labels**: Binary multi-label ground truth across 10 compound families (`basalt`, `carbonate`, `chloride`, `iron_oxide`, `oxalate`, `oxychlorine`, `phyllosilicate`, `silicate`, `sulfate`, `sulfide`) provided in `train_labels.csv` and `val_labels.csv`.
4. **Metadata & Packaging**: Standardized metadata (`metadata.csv`) with pseudonymous sample identifiers (`S0001`–`S0031`) and instrument assignments (`commercial` vs `sam_testbed`). No ground truth labels for the hidden test split are included in `/app/data`.
