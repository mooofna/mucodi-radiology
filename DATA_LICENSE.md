# Licences of the bundled scans

The web demo bundles three public chest CT scans and images derived from one of them. They are
redistributed under their original licences; the code licence (`LICENSE`) and the weights licence
(`WEIGHTS_LICENSE.md`) do not apply to them. Use of the scans must also follow the
[TCIA Data Usage Policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/).

No CT-RATE data, and nothing derived from CT-RATE images, is distributed: CT-RATE trained the
students, but its terms do not allow redistribution. The method figure therefore uses panels
re-rendered from the LIDC-IDRI case below. The evidence files hold aggregate results only, with no
per-case predictions or identifiers.

## Changes made to every bundled scan

The DICOM series were retrieved from the NCI Imaging Data Commons and converted to NIfTI with
dcm2niix. Each case ships as:

- a viewing volume in Hounsfield units, int16, reoriented to the closest canonical orientation,
  with the in-plane resolution halved (2×2 mean) when it exceeds 384 pixels;
- the model grid, the full field of view resized to 128×256×256 with trilinear interpolation and
  stored as Hounsfield units times 10 in int16;
- for `lidc-0001` only, the full-resolution volume in Hounsfield units, int16.

The method figure's eight scan panels are rendered from `lidc-0001` (resampled, randomly cropped
and windowed with the thesis's pipeline code).

## LIDC-IDRI: case `lidc-0001` (patient LIDC-IDRI-0001)

- Licence: [Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/)
  (CC BY 3.0).
- Acknowledgement: The authors acknowledge the National Cancer Institute and the Foundation for
  the National Institutes of Health, and their critical role in the creation of the free publicly
  available LIDC/IDRI Database used in this study.
- Data citation: Armato III, S. G., McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, C. R.,
  Reeves, A. P., et al. (2015). Data From LIDC-IDRI [Data set]. The Cancer Imaging Archive.
  <https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX>
- Publication: Armato, S. G. 3rd, McLennan, G., Bidaut, L., et al. (2011). The Lung Image
  Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference
  database of lung nodules on CT scans. Medical Physics, 38(2), 915-931.
  <https://doi.org/10.1118/1.3528204>

## MIDRC-RICORD-1A: cases `ricord-000082` and `ricord-000800`

Patients MIDRC-RICORD-1A-419639-000082 and MIDRC-RICORD-1A-419639-000800.

- Licence: [Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
  (CC BY-NC 4.0).
- Data citation: Tsai, E., Simpson, S., Lungren, M. P., Hershman, M., Roshkovan, L., Colak, E.,
  et al. (2020). Data from the Medical Imaging Data Resource Center (MIDRC), RSNA International
  COVID Radiology Database (RICORD) Release 1a, Chest CT Covid+ (MIDRC-RICORD-1A). The Cancer
  Imaging Archive. <https://doi.org/10.7937/VTW4-X588>
- Publication: Tsai, E. B., Simpson, S., Lungren, M. P., et al. (2021). The RSNA International
  COVID-19 Open Radiology Database (RICORD). Radiology, 299(1), E204-E213.
  <https://doi.org/10.1148/radiol.2021203957>

## Repositories

- The Cancer Imaging Archive: Clark, K., Vendt, B., Smith, K., et al. (2013). The Cancer Imaging
  Archive (TCIA): maintaining and operating a public information repository. Journal of Digital
  Imaging, 26(6), 1045-1057. <https://doi.org/10.1007/s10278-013-9622-7>
- NCI Imaging Data Commons: Fedorov, A., Longabaugh, W. J. R., Pot, D., et al. (2023). National
  Cancer Institute Imaging Data Commons: toward transparency, reproducibility, and scalability in
  imaging artificial intelligence. RadioGraphics, 43(12), e230180.
  <https://doi.org/10.1148/rg.230180>
