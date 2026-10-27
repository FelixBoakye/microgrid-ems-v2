# Data and Results License

## Scope

The **code** in this repository is licensed under the MIT License (see `LICENSE`).

The **generated results and data** produced by this codebase (result CSVs, tables,
figures, provenance manifests, and the synthesised demand profile) are licensed under the
**Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

- Legal code: https://creativecommons.org/licenses/by/4.0/legalcode
- Human-readable summary: https://creativecommons.org/licenses/by/4.0/

Covered by CC BY 4.0: `data/` (except the NASA POWER file, which keeps NASA's terms) and `results/`.

When reusing these results, please credit: Felix Sarkodie Boakye, University of Energy and Natural
Resources (UENR), Sunyani, Ghana, and cite the associated article (see `CITATION.cff`).

## Third-party data: NASA POWER

Hourly meteorological input (2018 to 2025) is obtained from the NASA POWER project. NASA
POWER data are not covered by the CC BY 4.0 grant above; they remain subject to NASA POWER's
own terms. Any NASA POWER data present in, or downloaded by, this repository is attributed as
follows.

### Acknowledgement (NASA POWER Referencing Guide wording)

NASA POWER asks that publications include both POWER's Reference and POWER's Data Reference:

> **POWER's Reference:** The data was obtained from National Aeronautics and Space
> Administration (NASA) Langley Research Center's Prediction Of Worldwide Energy Resources
> (POWER) project funded through the NASA Earth Science Division.

> **POWER's Data Reference:** The data was obtained from the POWER Project's Hourly
> NASA POWER hourly service (data covering 2018-01-01 to 2025-12-31 for the study site). Exact API version string was not recorded at download; the included CSV is the file used for the paper.

Source: NASA POWER Docs, "Referencing Guide", https://power.larc.nasa.gov/docs/referencing/
(accessed 2026-10-03).

Earlier (legacy) acknowledgement wording used by the POWER project, kept here for reference:
"These data were obtained from the NASA Langley Research Center POWER Project funded through
the NASA Earth Science Directorate Applied Science Program." (as quoted on the POWER Data
Access Viewer overview and in the `nasapower` R package documentation). The NASA Open Data
Portal entry (https://data.nasa.gov/dataset/prediction-of-worldwide-energy-resources-power)
gives: "This data was obtained from the NASA Langley Research Center (LaRC) POWER Project
funded through the NASA Earth Science/Applied Science Program." The current Referencing Guide
wording above takes precedence.

The POWER project also requests (Referencing Guide) that users send a reference or web link to
published work using POWER data to larc-power-project@mail.nasa.gov, and asks to be notified if
POWER data are transmitted to other researchers. This repository includes the NASA POWER hourly file used for the study; the associated article cites the source. Authors may notify NASA POWER at larc-power-project@mail.nasa.gov when a paper using the data is published.

## Cited, not redistributed

The following sources are **cited** in the paper and code documentation but are **not
redistributed** in this repository:

- Manufacturer diesel generator specification sheets used to fit fuel curves:
  Cummins C40D6, Generac SD040, Kohler 40REOZK. Obtain them from the respective manufacturers.
  Only the fitted curve coefficients (derived values) are included.
- Third-party demand statistics (Ghanaian and metered sub-Saharan sources) used to anchor the
  synthesised demand profile. See the paper's reference list for full citations.

If you believe any material in this repository is redistributed in error, please open an issue.
