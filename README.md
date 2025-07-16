# CGR-Diff

This repository contains the code for the CGR-Diff software. To run the software, you can use the following command:
```bash
python GUI.py
```
Alternatively, you can download and run the executable file from the following links.
- [CGR-Diff for Mac](https://drive.google.com/drive/folders/1PY2sN-PWIWRTAc-2o5lbrbgzJOZuFdOI?usp=sharing)
- [CGR-Diff for Linux](https://drive.google.com/file/d/11SWT93QyBsdzf1tOZsYQUEhPzfMPdX2T/view?usp=sharing)
- [CGR-Diff for Windows](https://drive.google.com/file/d/1F_tOTC_K3ocYcfrovsCrToGpQUMdHelC/view?usp=sharing)

You can also download the video tutorial from the following link:
- [CGR-Diff Video Tutorial](https://drive.google.com/file/d/1wTLiaFOS8Qjpv7w9OaGKkrYFIBQUda0n/view?usp=sharing)


## Dataset

Download the assemblies for each species from the NCBI website, and organize the data in the `Data` folder to match the following structure:
```
📂 Data/
├── 📂 Human/
│   ├── 📂 chromosomes/
│   │   ├── chr1.fna
│   │   └── ...
│   └── 📂 bedfiles/
│       ├── cytobands.bed
│       ├── telomere.bed
│       └── centromere.bed
├── 📂 Chimp/
│   └── 📂 chromosomes/
│       ├── chr1.fna
│       └── ...
├── 📂 Mouse/
│   └── 📂 chromosomes/
│       └── ....fna
├── 📂 Drosophila melanogaster/
│   └── 📂 chromosomes/
│       └── ....fna
├── 📂 Saccharomyces cerevisiae/
│   └── 📂 chromosomes/
│       └── ....fna
├── 📂 Arabidopsis thaliana/
│   └── 📂 chromosomes/
│       └── ....fna
├── 📂 Paramecium caudatum/
│   └── 📂 chromosomes/
│       └── ....fna (These are scaffolds)
├── 📂 Pyrococcus furiosus/
│   └── 📂 chromosomes/
│       └── ....fna
├── 📂 Escherichia coli/
│   └── 📂 chromosomes/
│       └── ....fna
├── 📂 Aspergillus nidulans/
│   └── 📂 chromosomes/
│       └── ....fna
├── 📂 Maize/
│   └── 📂 chromosomes/
│       └── ....fna
└── 📂 Dictyostelium discoideum/
    └── 📂 chromosomes/
        └── ....fna
```

Alternatively for `Human`, `Chimp`, `Mouse`, and `Maize` you can use the following command to download the chromosomes assemblies directly from the NCBI FTP server.

```bash
# Homo sapiens (human)
wget -P Data/Human/chromosomes/ -r -nH --cut-dirs=12 --no-parent \
ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/009/914/755/GCA_009914755.4_T2T-CHM13v2.0/GCA_009914755.4_T2T-CHM13v2.0_assembly_structure/Primary_Assembly/assembled_chromosomes/FASTA/
gunzip Data/Human/chromosomes/*.gz

# Pan troglodytes (chimpanzee)
wget -P Data/Chimp/chromosomes/ -r -nH --cut-dirs=12 --no-parent \
ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/028/858/775/GCA_028858775.2_NHGRI_mPanTro3-v2.0_pri/GCA_028858775.2_NHGRI_mPanTro3-v2.0_pri_assembly_structure/Primary_Assembly/assembled_chromosomes/FASTA/
gunzip Data/Chimp/chromosomes/*.gz

# Mus musculus (house mouse)
wget -P Data/Mouse/chromosomes/ -r -nH --cut-dirs=12 --no-parent \
ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/635/GCA_000001635.9_GRCm39/GCA_000001635.9_GRCm39_assembly_structure/Primary_Assembly/assembled_chromosomes/FASTA/
gunzip Data/Mouse/chromosomes/*.gz

# Zea mays (maize)
wget -P Data/Maize/chromosomes/ -r -nH --cut-dirs=12 --no-parent \
ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/022/117/705/GCA_022117705.1_Zm-Mo17-REFERENCE-CAU-T2T-assembly/GCA_022117705.1_Zm-Mo17-REFERENCE-CAU-T2T-assembly_assembly_structure/Primary_Assembly/assembled_chromosomes/FASTA/
gunzip Data/Maize/chromosomes/*.gz
```

You can also download the complete datasets and bedfiles used in the paper from the [Google Drive](https://drive.google.com/file/d/1q7fbymvlAd7XLA7D94QN575tON1qk1fR/view?usp=sharing).


The bedfiles in this dataset were processed from the original CHM13 dataset provided by the [CHM13 GitHub repository](https://github.com/marbl/CHM13). However, in the `cytobands.bed` file, the color of each cytoband region is added based on the [NCBI Genome Data Viewer](https://www.ncbi.nlm.nih.gov/gdv/browser/genome/?id=GCF_009914755.1).
Please cite both the original dataset and this repository when using this processed dataset.
