# FastNet - Version ?? - TRL2 - 2026-05-07

## Versioning and Reference Information

### Card Authors  
Sophie Arana

### Model/System Name  
<!-- Unique, recognizable name for this model/family of models. -->
IceNet Multimodal Pipeline (icenet-mp)

### Version  
tbd

### TRL  
TRL 2

### Model/System Release Date  
tbd

### TRL Date  
fill in once reviewed by team

### TRL Card Date  
fill in once reviewed by team

### Lead  
Louisa van Zeeland

### Contact Information 
Louisa van Zeeland

### Access to Products  
open source

### Licence  
MIT Licensed codebase

### References  

### Citation  
The Alan Turing Institute. (2026). IceNet Multimodal Pipeline [Source code]. Github. https://github.com/alan-turing-institute/icenet-mp

## Model/System Details

### Description  
An end-to-end multimodal AI forecasting system for Arctic and Antarctic sea ice, integrating satellite imagery, reanalysis data, and point-based data (in-situ sensor data) to deliver reliable probabilistic predictions across short-term timescales. Designed to support critical decision-making for stakeholders ranging from Arctic shipping and indigenous communities to wildlife protection agencies. 
==add description on model architectures==

### Intended Uses  
❌ **What is the consequence** of adding something as "intended use"? what is the required evidence?
- Arctic and Antarctic sea ice forecasting 

### Out-of-Scope Uses  
- Non-polar regions
- Non-sea-ice environmental forecasting (e.g. land, freshwater ice)
- Real-time operations
- variables beyond SIC

### Time Lag  
❌ this time lag should ideally be benchmarked and only the relevant time lags should be reported.
**Short-term** forecasting (operational/near-term predictions)
With particular focus on critical zones like the **sea ice edge** and **marginal ice zone**.

### Spatial Domain  
global (Arctic and Antarctic)

### Temporal / Seasonal Domain  
<!-- Valid time periods, seasons, training temporal coverage. -->
valid time periods
❌ Need to have benchmarking for seasonal model performance

###  s
<!-- Methods for uncertainty/variability in inputs and outputs. -->
==should be methods for quantifying?==

## Training, Testing, Validation Datasets and Procedures

### Input Information  
<!-- Data sources, sensors, processing, rationale, training set construction,
data leakage avoidance, possible duplicates/overlapping, etc. -->
- **Data sources:** OSI-SAF L4 derived SIC product and ERA5 reanalysis data, processed and normalised using Anemoi dataset tooling
- **Data leakage:** Strict temporal separation enforced, with test years held out exclusively for benchmarking and never exposed during development
- **Input uncertainty:** Missing data identified and handled via Anemoi's built-in inspection tooling; full formalisation of input uncertainty remains outstanding
- **Output uncertainty:** Currently no default uncertainty quantification. The DDPM architecture supports epistemic uncertainty estimation through multiple sampling runs, producing a spread of plausible SIC forecasts rather than a single deterministic prediction

==add input data table==

### Output Sea Ice Variables  
<!--
List of abbreviations for sea ice output variables.
(Refer to: JCOMM TR 80 Ice Objects Catalogue)
-->
SIC in % covered by ice per grid cell, with 0% being open water and 100% being full ice cover

### Output Information  
<!-- Description of outputs, formats, structure, resolution. -->
❌ currently numpy arrays but output is never saved, should include this as an optional flag to be able to communicate what outputs are possible to end users

## Evaluation Details

### Evaluation Statement  
Evaluation metrics: `sieerror (mean)`, `rmse (mean)`, `mae (mean)`
Model comparison: simple unet, persistence
threshold - 15% sea ice for ice edge
==need to add proper report against what benchmarks results have been compared==
<!-- Qualitative/quantitative evaluations: metrics, key findings, comparison to other models, calibration/threshold factors and their determination/performance. -->

### Rationale for Current TRL  
The icenet-mp system has made substantial progress through TRL 2, with optimised code, unit tests, an initial curated datasets for in-situ data, and a basic software architecture established. Notably, several TRL 3 coding checkpoints have already been completed, including modular/reusable code structures and ==integration tests for module and API robustness==. 

### On-going Progress Towards Next TRL  
<!-- Key elements being addressed for next TRL (checklist items, internal workplans, reasons for pause/halt if applicable). -->
To complete TRL 2, there are two outstanding requirements: formal documentation of baseline model performance metrics and fully validated focused experiments confirming model behaviours and goals.

### Ethical Considerations  
<!-- Sensitive data use, decision-making requirements, risks or misuse, problematic cases, ownership/community considerations. -->
The Icenet-multimodal pipeline is a research product and currently operates on publicly accessible data so there are no known ethical considerations.
