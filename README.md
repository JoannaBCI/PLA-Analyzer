Proximity Ligation Assay (PLA) image analyzer.
Quantifies protein-protein interaction puncta (dots) per nucleus from confocal fluorescence microscopy images.
 
Protocol based on:

Lopez-Cano et al. (2019) "Proximity Ligation Assay Image Analysis Protocol:
  Addressing Receptor-Receptor Interactions." Methods in Molecular Biology.
 
Inputs:
  --dapi Path to DAPI channel image (nuclei), TIFF/PNG/BMP
  
  --pla  Path to PLA channel image (fluorescent dots), TIFF/PNG/BMP
 
Outputs (saved next to the DAPI file in a new folder):

  - overlay image: nuclei outlines + PLA dots colored by nucleus
    
  - bar plot: dots/nucleus ratio
    
  - CSV: per-nucleus counts
 
Usage: python pla_analyzer.py --dapi DAPI.tif --pla PLA.tif

Dependencies:  pip install numpy pandas matplotlib scikit-image tifffile scipy
